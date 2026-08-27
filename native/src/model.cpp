#include "shadow/model.hpp"
#include "shadow/archive.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#if defined(__APPLE__)
#include <pthread.h>
#include <sys/sysctl.h>
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace shadowrt {
namespace {

constexpr std::size_t D = 1536, L = 10, NH = 24, NKV = 2, HD = 64, FPD = 512;

struct ProfileCounters {
  double dense = 0, rvq = 0, ternary = 0, logits = 0, attention = 0, other = 0;
  double embedding = 0, qkv = 0, output = 0, ffn_up_gate = 0, ffn_down = 0;
  double structural = 0, head = 0;
  std::uint64_t dense_calls = 0, rvq_calls = 0, ternary_calls = 0;
};

ProfileCounters operator-(const ProfileCounters &left,
                          const ProfileCounters &right) {
  ProfileCounters result;
#define SHADOW_PROFILE_SUBTRACT(field) result.field = left.field - right.field
  SHADOW_PROFILE_SUBTRACT(dense);
  SHADOW_PROFILE_SUBTRACT(rvq);
  SHADOW_PROFILE_SUBTRACT(ternary);
  SHADOW_PROFILE_SUBTRACT(logits);
  SHADOW_PROFILE_SUBTRACT(attention);
  SHADOW_PROFILE_SUBTRACT(embedding);
  SHADOW_PROFILE_SUBTRACT(qkv);
  SHADOW_PROFILE_SUBTRACT(output);
  SHADOW_PROFILE_SUBTRACT(ffn_up_gate);
  SHADOW_PROFILE_SUBTRACT(ffn_down);
  SHADOW_PROFILE_SUBTRACT(structural);
  SHADOW_PROFILE_SUBTRACT(head);
  SHADOW_PROFILE_SUBTRACT(dense_calls);
  SHADOW_PROFILE_SUBTRACT(rvq_calls);
  SHADOW_PROFILE_SUBTRACT(ternary_calls);
#undef SHADOW_PROFILE_SUBTRACT
  return result;
}
thread_local ProfileCounters *active_profile = nullptr;

struct ProfileScope {
  ProfileScope(ProfileCounters *counters) : previous(active_profile) {
    active_profile = counters;
  }
  ~ProfileScope() { active_profile = previous; }
  ProfileCounters *previous;
};

template <class T> T read(std::istream &in) {
  T value{};
  in.read(reinterpret_cast<char *>(&value), sizeof(value));
  if (!in)
    throw std::runtime_error("truncated binary file");
  return value;
}

std::vector<std::uint8_t> read_bytes(std::istream &in, std::size_t n) {
  std::vector<std::uint8_t> result(n);
  in.read(reinterpret_cast<char *>(result.data()),
          static_cast<std::streamsize>(n));
  if (!in)
    throw std::runtime_error("truncated binary file");
  return result;
}

void dump_ffn_activation(std::string_view stage, std::size_t layer,
                         std::span<const float> values) {
  const char *path = std::getenv("SHADOW_DUMP_FFN_ACTIVATIONS");
  if (!path || !*path)
    return;
  static std::mutex mutex;
  std::lock_guard lock(mutex);
  std::ofstream out(path, std::ios::binary | std::ios::app);
  if (!out)
    throw std::runtime_error("cannot append FFN activation dump");
  const std::uint32_t magic = 0x31414653; // SFA1
  const std::uint32_t stage_id = stage == "up" ? 1 : 0;
  const std::uint32_t layer_id = static_cast<std::uint32_t>(layer);
  const std::uint32_t count = static_cast<std::uint32_t>(values.size());
  out.write(reinterpret_cast<const char *>(&magic), sizeof(magic));
  out.write(reinterpret_cast<const char *>(&stage_id), sizeof(stage_id));
  out.write(reinterpret_cast<const char *>(&layer_id), sizeof(layer_id));
  out.write(reinterpret_cast<const char *>(&count), sizeof(count));
  out.write(reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size_bytes()));
}

void write_npy_logits(const std::filesystem::path &path,
                      std::span<const std::vector<float>> rows) {
  if (rows.empty())
    return;
  const std::size_t columns = rows.front().size();
  for (const auto &row : rows)
    if (row.size() != columns)
      throw std::runtime_error("logit dump row mismatch");
  std::string dictionary =
      "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
      std::to_string(rows.size()) + ", " + std::to_string(columns) + "), }";
  const std::size_t preamble = 10;
  const std::size_t padding =
      (64 - ((preamble + dictionary.size() + 1) % 64)) % 64;
  dictionary.append(padding, ' ');
  dictionary.push_back('\n');
  std::ofstream out(path, std::ios::binary);
  if (!out)
    throw std::runtime_error("cannot create logit dump: " + path.string());
  out.write("\x93NUMPY", 6);
  const std::uint8_t version[2] = {1, 0};
  out.write(reinterpret_cast<const char *>(version), 2);
  const auto length = static_cast<std::uint16_t>(dictionary.size());
  out.write(reinterpret_cast<const char *>(&length), 2);
  out.write(dictionary.data(), dictionary.size());
  for (const auto &row : rows)
    out.write(reinterpret_cast<const char *>(row.data()),
              static_cast<std::streamsize>(row.size() * sizeof(float)));
}

std::size_t product(const std::vector<std::uint32_t> &dims) {
  return std::accumulate(dims.begin(), dims.end(), std::size_t{1},
                         std::multiplies<>());
}

std::size_t thread_count() {
  if (const char *value = std::getenv("SHADOW_THREADS"))
    return std::max(1ul, std::stoul(value));
#if defined(__APPLE__)
  std::uint32_t performance_cores = 0;
  std::size_t size = sizeof(performance_cores);
  if (sysctlbyname("hw.perflevel0.physicalcpu", &performance_cores, &size,
                   nullptr, 0) == 0 &&
      performance_cores > 0)
    return performance_cores;
#endif
  return std::max(1u, std::thread::hardware_concurrency());
}

class WorkerPool {
public:
  explicit WorkerPool(std::size_t threads)
      : threads_(std::max<std::size_t>(1, threads)) {
    workers_.reserve(threads_ - 1);
    for (std::size_t worker = 1; worker < threads_; ++worker)
      workers_.emplace_back([this, worker] { worker_loop(worker); });
  }

  ~WorkerPool() {
    stopping_.store(true, std::memory_order_release);
    generation_.fetch_add(1, std::memory_order_release);
    for (auto &worker : workers_)
      worker.join();
  }

  void run(std::size_t rows,
           std::function<void(std::size_t, std::size_t)> task) {
    if (threads_ == 1) {
      task(0, rows);
      return;
    }
    rows_ = rows;
    task_ = std::move(task);
    completed_.store(0, std::memory_order_relaxed);
    generation_.fetch_add(1, std::memory_order_release);
    task_(0, boundary(1));
    while (completed_.load(std::memory_order_acquire) != threads_ - 1)
      __asm__ volatile("yield");
  }

private:
  std::size_t boundary(std::size_t worker) const {
    if (worker == 0)
      return 0;
    if (worker >= threads_)
      return rows_;
    const std::size_t raw = rows_ * worker / threads_;
    return std::min(rows_, (raw + 15) & ~std::size_t{15});
  }

  void worker_loop(std::size_t worker) {
#if defined(__APPLE__)
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    std::uint64_t seen = 0;
    for (;;) {
      std::uint64_t current;
      while ((current = generation_.load(std::memory_order_acquire)) == seen)
        __asm__ volatile("yield");
      if (stopping_.load(std::memory_order_acquire))
        return;
      task_(boundary(worker), boundary(worker + 1));
      completed_.fetch_add(1, std::memory_order_release);
      seen = current;
    }
  }

  std::size_t threads_, rows_ = 0;
  std::atomic<bool> stopping_{false};
  std::atomic<std::uint64_t> generation_{0};
  std::atomic<std::size_t> completed_{0};
  std::function<void(std::size_t, std::size_t)> task_;
  std::vector<std::thread> workers_;
};

WorkerPool &worker_pool() {
  static WorkerPool pool(thread_count());
  return pool;
}

template <class F> void parallel_rows(std::size_t rows, F &&fn) {
  if (rows < 64 || thread_count() == 1) {
    fn(0, rows);
    return;
  }
  worker_pool().run(rows, std::forward<F>(fn));
}

float half_to_float(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000) << 16;
  std::uint32_t exp = (h >> 10) & 0x1f, mant = h & 0x3ff, bits;
  if (exp == 0) {
    if (mant == 0)
      bits = sign;
    else {
      exp = 127 - 15 + 1;
      while ((mant & 0x400) == 0) {
        mant <<= 1;
        --exp;
      }
      bits = sign | (exp << 23) | ((mant & 0x3ff) << 13);
    }
  } else if (exp == 31)
    bits = sign | 0x7f800000 | (mant << 13);
  else
    bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  return std::bit_cast<float>(bits);
}

float bfloat16_round(float value) {
  std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  if ((bits & 0x7f800000u) == 0x7f800000u)
    return value;
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return std::bit_cast<float>(bits & 0xffff0000u);
}

void bfloat16_round_inplace(std::span<float> values) {
  for (float &value : values)
    value = bfloat16_round(value);
}

const auto &ternary3_table() {
  static const auto table = [] {
    std::array<std::array<std::int8_t, 5>, 256> result{};
    for (std::size_t byte = 0; byte < result.size(); ++byte) {
      unsigned value = static_cast<unsigned>(byte);
      for (auto &code : result[byte]) {
        code = static_cast<std::int8_t>(value % 3) - 1;
        value /= 3;
      }
    }
    return result;
  }();
  return table;
}

const auto &fingerprint_sign_table() {
  static const auto table = [] {
    std::array<std::array<std::int8_t, 8>, 256> result{};
    for (std::size_t byte = 0; byte < result.size(); ++byte)
      for (std::size_t bit = 0; bit < 8; ++bit)
        result[byte][bit] = byte & (1u << (7 - bit)) ? 1 : -1;
    return result;
  }();
  return table;
}

bool ternary_reference_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("SHADOW_TERNARY_REFERENCE");
    return value && std::string_view(value) != "0";
  }();
  return enabled;
}

bool logits_reference_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("SHADOW_LOGITS_REFERENCE");
    return value && std::string_view(value) != "0";
  }();
  return enabled;
}

bool fast_logits_enabled() {
  static const bool enabled = [] {
    const char *value = std::getenv("SHADOW_FAST_LOGITS");
    return value && std::string_view(value) != "0";
  }();
  return enabled;
}

std::size_t dotprod_ffn_group() {
#if defined(SHADOW_ARM_DOTPROD)
  static const std::size_t group = [] {
    const char *value = std::getenv("SHADOW_DOTPROD_FFN");
    if (!value || !*value || std::string_view(value) == "0")
      return std::size_t{0};
    if (std::string_view(value) == "compact64")
      return std::size_t{64};
    const std::size_t parsed = std::stoul(value);
    if (parsed != 64 && parsed != 128)
      throw std::runtime_error("SHADOW_DOTPROD_FFN must be 64 or 128");
    return parsed;
  }();
  return group;
#else
  return 0;
#endif
}

bool compact_dotprod_ffn() {
#if defined(SHADOW_ARM_DOTPROD)
  const char *value = std::getenv("SHADOW_DOTPROD_FFN");
  return value && std::string_view(value) == "compact64";
#else
  return false;
#endif
}

void silu_inplace(std::span<float> values) {
  for (float &value : values)
    value = value / (1.0f + std::exp(-value));
}

void silu_multiply_inplace(std::span<float> values, std::span<const float> gate) {
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] *= gate[i] / (1.0f + std::exp(-gate[i]));
}

std::size_t ternary3_offset(std::size_t row, std::size_t block,
                            std::size_t stride) {
  return ((row / 8) * stride + block) * 8 + row % 8;
}

std::size_t ternary_nibble_offset(std::size_t row, std::size_t column,
                                  std::size_t input_width) {
  return ((row / 16) * input_width + column) * 8;
}

std::size_t ternary_dotprod_offset(std::size_t row, std::size_t column,
                                   std::size_t input_width) {
  return ((row / 16) * (input_width / 4) + column / 4) * 64 +
         (row % 16) * 4 + column % 4;
}

std::size_t ternary_dotprod_compact_offset(std::size_t row, std::size_t column,
                                           std::size_t input_width) {
  return ((row / 16) * (input_width / 4) + column / 4) * 16 + row % 16;
}

#if defined(__aarch64__)
int8x16_t unpack_ternary_nibbles(const std::uint8_t *packed) {
  const int8x8_t bytes = vreinterpret_s8_u8(vld1_u8(packed));
  const int8x8_t low = vshr_n_s8(vshl_n_s8(bytes, 4), 4);
  const int8x8_t high = vshr_n_s8(bytes, 4);
  return vcombine_s8(low, high);
}
#endif

std::size_t fingerprint_offset(std::size_t row, std::size_t byte) {
  return ((row / 8) * 64 + byte) * 8 + row % 8;
}

float dot(std::span<const float> a, std::span<const float> b) {
  if (a.size() != b.size())
    throw std::runtime_error("dot shape mismatch");
#if defined(__aarch64__)
  float32x4_t sum = vdupq_n_f32(0.0f);
  std::size_t i = 0;
  for (; i + 4 <= a.size(); i += 4)
    sum = vfmaq_f32(sum, vld1q_f32(a.data() + i), vld1q_f32(b.data() + i));
  float result = vaddvq_f32(sum);
  for (; i < a.size(); ++i)
    result += a[i] * b[i];
  return result;
#else
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
#endif
}

void add_inplace(std::span<float> a, std::span<const float> b) {
  for (std::size_t i = 0; i < a.size(); ++i)
    a[i] += b[i];
}

void rms_into(std::span<const float> x, const Tensor &weight,
              std::span<float> out) {
  if (out.size() != x.size())
    throw std::runtime_error("RMS output shape mismatch");
  const auto w = weight.dense_f32();
  const float scale =
      1.0f / std::sqrt(dot(x, x) / static_cast<float>(x.size()) + 1e-6f);
  for (std::size_t i = 0; i < x.size(); ++i)
    out[i] = x[i] * scale * w[i];
}

void pot_inplace(std::span<float> x) {
  float m = 1e-6f;
  for (float v : x)
    m = std::max(m, std::abs(v));
  const float scale = std::exp2(std::ceil(std::log2(m / 127.0f)));
  for (float &v : x)
    v = std::clamp(std::nearbyint(v / scale), -127.0f, 127.0f) * scale;
}

void rope_inplace(std::span<float> x, std::uint64_t position) {
  for (std::size_t i = 0; i < HD; i += 2) {
    const float angle = static_cast<float>(position) /
                        std::pow(10000.0f, static_cast<float>(i) / HD);
    const float c = bfloat16_round(std::cos(angle)),
                s = bfloat16_round(std::sin(angle)), a = x[i], b = x[i + 1];
    x[i] = a * c - b * s;
    x[i + 1] = a * s + b * c;
  }
}

void walsh_hadamard(std::span<float> x) {
  for (std::size_t width = 1; width < x.size(); width *= 2)
    for (std::size_t base = 0; base < x.size(); base += 2 * width)
      for (std::size_t j = 0; j < width; ++j) {
        const float a = x[base + j], b = x[base + width + j];
        x[base + j] = a + b;
        x[base + width + j] = a - b;
      }
  const float scale = 1.0f / std::sqrt(static_cast<float>(x.size()));
  for (float &value : x)
    value *= scale;
}

std::array<std::uint8_t, HD / 8> codec_pack(std::span<const float> x,
                                            const ModelFile &model,
                                            const std::string &prefix,
                                            std::size_t head) {
  const auto sign = model.at(prefix + "sign").dense_f32(),
             mu = model.at(prefix + "mu").dense_f32();
  const auto ctv = model.at(prefix + "ctv").dense_f32();
  const std::size_t offset = head * HD;
  std::array<float, HD> rotated{};
  for (std::size_t i = 0; i < HD; ++i)
    rotated[i] = (x[i] - mu[offset + i]) * sign[offset + i];
  walsh_hadamard(rotated);
  std::array<std::uint8_t, HD / 8> packed{};
  for (std::size_t i = 0; i < HD; ++i) {
    const float decision = std::nearbyint(rotated[i] * 256.0f) / 256.0f;
    const float threshold = std::nearbyint(ctv[offset + i] * 256.0f) / 256.0f;
    if (decision > threshold)
      packed[i / 8] |= static_cast<std::uint8_t>(1u << (i % 8));
  }
  return packed;
}

std::array<float, HD> codec_unpack(std::span<const std::uint8_t> packed,
                                   const ModelFile &model,
                                   const std::string &prefix,
                                   std::size_t head) {
  const auto sign = model.at(prefix + "sign").dense_f32(),
             mu = model.at(prefix + "mu").dense_f32();
  const auto low = model.at(prefix + "low").dense_f32(),
             high = model.at(prefix + "high").dense_f32();
  const std::size_t offset = head * HD;
  std::array<float, HD> rotated{};
  for (std::size_t i = 0; i < HD; ++i)
    rotated[i] =
        (packed[i / 8] >> (i % 8)) & 1 ? high[offset + i] : low[offset + i];
  walsh_hadamard(rotated);
  for (std::size_t i = 0; i < HD; ++i)
    rotated[i] = mu[offset + i] + rotated[i] * sign[offset + i];
  return rotated;
}

struct LayerCache {
  std::array<std::vector<std::array<float, HD>>, NKV> keys, values;
};

struct LayerWeights {
  const Tensor *n1, *n2, *q, *k, *v, *o, *qn, *kn, *g, *alpha, *up, *gt, *dn;
  const Tensor *k_sign = nullptr, *k_mu = nullptr, *k_ctv = nullptr,
               *k_low = nullptr, *k_high = nullptr;
  const Tensor *v_sign = nullptr, *v_mu = nullptr, *v_ctv = nullptr,
               *v_low = nullptr, *v_high = nullptr;
};

std::size_t argmax_logits(std::span<const float> logits) {
  if (logits.empty())
    throw std::runtime_error("cannot sample empty logits");
#if defined(__aarch64__)
  float32x4_t maximum = vdupq_n_f32(-std::numeric_limits<float>::infinity());
  std::size_t i = 0;
  for (; i + 4 <= logits.size(); i += 4)
    maximum = vmaxq_f32(maximum, vld1q_f32(logits.data() + i));
  float peak = vmaxvq_f32(maximum);
  for (; i < logits.size(); ++i)
    peak = std::max(peak, logits[i]);
  for (std::size_t index = 0; index < logits.size(); ++index)
    if (logits[index] == peak)
      return index;
#endif
  return static_cast<std::size_t>(
      std::max_element(logits.begin(), logits.end()) - logits.begin());
}

std::size_t sample_token(std::vector<float> &logits,
                         const GenerationOptions &options,
                         std::span<const std::uint32_t> history,
                         std::mt19937_64 &rng) {
  if (options.repetition_penalty != 1.0f) {
    for (auto id : history)
      if (id < logits.size())
        logits[id] = logits[id] < 0 ? logits[id] * options.repetition_penalty
                                    : logits[id] / options.repetition_penalty;
  }
  if (options.temperature <= 0.0f || options.top_k <= 1)
    return argmax_logits(logits);
  const std::size_t k = std::min(options.top_k, logits.size());
  std::vector<std::size_t> ids(logits.size());
  std::iota(ids.begin(), ids.end(), 0);
  std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
                    [&](auto a, auto b) { return logits[a] > logits[b]; });
  const float peak = logits[ids[0]] / options.temperature;
  std::vector<double> weights(k);
  for (std::size_t i = 0; i < k; ++i)
    weights[i] = std::exp(
        static_cast<double>(logits[ids[i]] / options.temperature - peak));
  return ids[std::discrete_distribution<std::size_t>(weights.begin(),
                                                     weights.end())(rng)];
}

} // namespace

std::span<const float> Tensor::dense_f32() const {
  if (kind != WeightKind::dense_f32)
    throw std::runtime_error(name + " is not float32");
  return {reinterpret_cast<const float *>(bytes.data()),
          bytes.size() / sizeof(float)};
}

std::vector<float> Tensor::matvec(std::span<const float> x) const {
  std::vector<float> y(out);
  matvec_into(x, y);
  return y;
}

void Tensor::matvec_into(std::span<const float> x, std::span<float> y) const {
  const auto started = active_profile ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
  if (x.size() != in)
    throw std::runtime_error(name + " matvec input mismatch");
  if (y.size() != out)
    throw std::runtime_error(name + " matvec output mismatch");
  std::fill(y.begin(), y.end(), 0.0f);
  if (kind == WeightKind::dense_f32) {
    const auto w = dense_f32();
    parallel_rows(out, [&](std::size_t begin, std::size_t end) {
      for (std::size_t r = begin; r < end; ++r)
        y[r] = dot(x, w.subspan(r * in, in));
    });
  } else if (kind == WeightKind::dense_f16) {
    const auto *w = reinterpret_cast<const std::uint16_t *>(bytes.data());
    parallel_rows(out, [&](std::size_t begin, std::size_t end) {
      for (std::size_t r = begin; r < end; ++r) {
#if defined(__aarch64__)
        float32x4_t sum = vdupq_n_f32(0);
        std::size_t c = 0;
        const auto *row = reinterpret_cast<const float16_t *>(w + r * in);
        for (; c + 4 <= in; c += 4)
          sum = vfmaq_f32(sum, vld1q_f32(x.data() + c),
                          vcvt_f32_f16(vld1_f16(row + c)));
        y[r] = vaddvq_f32(sum);
        for (; c < in; ++c)
          y[r] += x[c] * half_to_float(w[r * in + c]);
#else
                for (std::size_t c = 0; c < in; ++c) y[r] += x[c] * half_to_float(w[r * in + c]);
#endif
      }
    });
  } else if (kind == WeightKind::ternary3) {
#if defined(SHADOW_ARM_DOTPROD) && defined(__aarch64__)
    const std::size_t quant_group = dotprod_ffn_group();
    if (quant_group && (!ternary_dotprod.empty() ||
                        !ternary_dotprod_compact.empty())) {
      std::vector<std::int8_t> quantized;
      std::vector<float> quant_scales;
      quantized.resize(in);
      quant_scales.resize((in + quant_group - 1) / quant_group);
      for (std::size_t begin = 0; begin < in; begin += quant_group) {
        const std::size_t end = std::min(begin + quant_group, std::size_t(in));
        float peak = 0;
        for (std::size_t column = begin; column < end; ++column)
          peak = std::max(peak, std::abs(x[column]));
        const float scale = peak / 127.0f;
        const float inverse = scale == 0 ? 0 : 1 / scale;
        quant_scales[begin / quant_group] = scale;
        for (std::size_t column = begin; column < end; ++column)
          quantized[column] = static_cast<std::int8_t>(std::clamp(
              std::nearbyint(x[column] * inverse), -127.0f, 127.0f));
      }
      parallel_rows(out, [&](std::size_t row_begin, std::size_t row_end) {
        for (std::size_t row = row_begin; row < row_end; row += 16) {
          float32x4_t output[4];
          for (auto &part : output) part = vdupq_n_f32(0);
          for (std::size_t begin = 0; begin < in; begin += quant_group) {
            int32x4_t sums[4];
            for (auto &sum : sums) sum = vdupq_n_s32(0);
            const std::size_t end = std::min(begin + quant_group, std::size_t(in));
            for (std::size_t column = begin; column < end; column += 4) {
              std::int32_t word;
              std::memcpy(&word, quantized.data() + column, sizeof(word));
              const int8x16_t activation =
                  vreinterpretq_s8_s32(vdupq_n_s32(word));
              if (!ternary_dotprod_compact.empty()) {
                const uint8x16_t packed = vld1q_u8(
                    ternary_dotprod_compact.data() +
                    ternary_dotprod_compact_offset(row, column, in));
                const uint8x16_t mask = vdupq_n_u8(3), one = vdupq_n_u8(1);
                const int8x16_t q0 = vreinterpretq_s8_u8(
                    vsubq_u8(vandq_u8(packed, mask), one));
                const int8x16_t q1 = vreinterpretq_s8_u8(vsubq_u8(
                    vandq_u8(vshrq_n_u8(packed, 2), mask), one));
                const int8x16_t q2 = vreinterpretq_s8_u8(vsubq_u8(
                    vandq_u8(vshrq_n_u8(packed, 4), mask), one));
                const int8x16_t q3 = vreinterpretq_s8_u8(
                    vsubq_u8(vshrq_n_u8(packed, 6), one));
                const auto z01 = vzipq_s8(q0, q1), z23 = vzipq_s8(q2, q3);
                const auto rows0 = vzipq_s16(vreinterpretq_s16_s8(z01.val[0]),
                                             vreinterpretq_s16_s8(z23.val[0]));
                const auto rows1 = vzipq_s16(vreinterpretq_s16_s8(z01.val[1]),
                                             vreinterpretq_s16_s8(z23.val[1]));
                const int8x16_t tile[4] = {
                    vreinterpretq_s8_s16(rows0.val[0]),
                    vreinterpretq_s8_s16(rows0.val[1]),
                    vreinterpretq_s8_s16(rows1.val[0]),
                    vreinterpretq_s8_s16(rows1.val[1])};
                for (int lane = 0; lane < 4; ++lane)
                  sums[lane] = vdotq_s32(sums[lane], tile[lane], activation);
              } else {
                const auto *weights = ternary_dotprod.data() +
                    ternary_dotprod_offset(row, column, in);
                for (int lane = 0; lane < 4; ++lane)
                  sums[lane] = vdotq_s32(
                      sums[lane], vld1q_s8(weights + lane * 16), activation);
              }
            }
            const float scale = quant_scales[begin / quant_group];
            for (int lane = 0; lane < 4; ++lane)
              output[lane] = vfmaq_n_f32(
                  output[lane], vcvtq_f32_s32(sums[lane]), scale);
          }
          float values[16];
          for (int lane = 0; lane < 4; ++lane)
            vst1q_f32(values + lane * 4, output[lane]);
          for (std::size_t lane = 0; lane < 16; ++lane)
            y[row + lane] = values[lane] * scales[row + lane];
        }
      });
      return;
    }
#endif
    const std::size_t stride = (in + 4) / 5;
    const bool reference = ternary_reference_enabled();
    parallel_rows(out, [&](std::size_t begin, std::size_t end) {
      const float *__restrict input_values = x.data();
      float *__restrict output_values = y.data();
      const std::uint8_t *__restrict expanded_weights = ternary_nibbles.data();
      const float *__restrict row_scales = scales.data();
      if (reference) {
        for (std::size_t r = begin; r < end; ++r) {
          float sum = 0.0f;
          for (std::size_t block = 0; block < stride; ++block) {
            const auto &codes =
                ternary3_table()[bytes[ternary3_offset(r, block, stride)]];
            for (std::size_t j = 0, c = block * 5; j < 5 && c + j < in; ++j) {
              if (codes[j] < 0)
                sum -= x[c + j];
              else if (codes[j] > 0)
                sum += x[c + j];
            }
          }
          y[r] = sum * scales[r];
        }
        return;
      }
      const std::size_t complete = in / 5;
      std::size_t r = begin;
      for (; r < end && r % 16 != 0; ++r) {
        float sum = 0.0f;
        for (std::size_t block = 0; block < complete; ++block) {
          const auto &codes =
              ternary3_table()[bytes[ternary3_offset(r, block, stride)]];
          const float *input = x.data() + block * 5;
          sum += input[0] * codes[0];
          sum += input[1] * codes[1];
          sum += input[2] * codes[2];
          sum += input[3] * codes[3];
          sum += input[4] * codes[4];
        }
        if (complete < stride) {
          const auto &codes =
              ternary3_table()[bytes[ternary3_offset(r, complete, stride)]];
          for (std::size_t c = complete * 5; c < in; ++c)
            sum += x[c] * codes[c - complete * 5];
        }
        y[r] = sum * scales[r];
      }
#if defined(__aarch64__)
      for (; r + 16 <= end; r += 16) {
        float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0),
                    b0 = vdupq_n_f32(0), b1 = vdupq_n_f32(0);
#if defined(__clang__)
#pragma clang loop unroll_count(2)
#endif
        for (std::size_t column = 0; column < in; ++column) {
          const int8x16_t codes = unpack_ternary_nibbles(
              expanded_weights + ternary_nibble_offset(r, column, in));
          const int16x8_t lo = vmovl_s8(vget_low_s8(codes)),
                          hi = vmovl_s8(vget_high_s8(codes));
          a0 = vfmaq_n_f32(a0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),
                           input_values[column]);
          a1 = vfmaq_n_f32(a1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))),
                           input_values[column]);
          b0 = vfmaq_n_f32(b0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),
                           input_values[column]);
          b1 = vfmaq_n_f32(b1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))),
                           input_values[column]);
        }
        float sums[16];
        vst1q_f32(sums, a0);
        vst1q_f32(sums + 4, a1);
        vst1q_f32(sums + 8, b0);
        vst1q_f32(sums + 12, b1);
        for (std::size_t row = 0; row < 16; ++row)
          output_values[r + row] = sums[row] * row_scales[r + row];
      }
#endif
      for (; r + 8 <= end; r += 8) {
#if defined(__aarch64__)
        float32x4_t sums_lo = vdupq_n_f32(0.0f), sums_hi = vdupq_n_f32(0.0f);
        for (std::size_t block = 0; block < complete; ++block) {
          const float *input = x.data() + block * 5;
          uint16x8_t packed = vmovl_u8(
              vld1_u8(bytes.data() + ternary3_offset(r, block, stride)));
          for (std::size_t j = 0; j < 5; ++j) {
            const uint16x8_t quotient =
                vshrq_n_u16(vmulq_n_u16(packed, 171), 9);
            const int16x8_t digit = vreinterpretq_s16_u16(
                vsubq_u16(packed, vmulq_n_u16(quotient, 3)));
            const int16x8_t signed_digit = vsubq_s16(digit, vdupq_n_s16(1));
            const float32x4_t low =
                vcvtq_f32_s32(vmovl_s16(vget_low_s16(signed_digit)));
            const float32x4_t high =
                vcvtq_f32_s32(vmovl_s16(vget_high_s16(signed_digit)));
            sums_lo = vfmaq_n_f32(sums_lo, low, input[j]);
            sums_hi = vfmaq_n_f32(sums_hi, high, input[j]);
            packed = quotient;
          }
        }
        float sums[8];
        vst1q_f32(sums, sums_lo);
        vst1q_f32(sums + 4, sums_hi);
#else
                float sums[8]{};
                for (std::size_t block = 0; block < complete; ++block) {
                    const float* input = x.data() + block * 5;
                    const auto* packed = bytes.data() + ternary3_offset(r, block, stride);
                    const auto* c0 = &ternary3_table()[packed[0]];
                    const auto* c1 = &ternary3_table()[packed[1]];
                    const auto* c2 = &ternary3_table()[packed[2]];
                    const auto* c3 = &ternary3_table()[packed[3]];
                    const auto* c4 = &ternary3_table()[packed[4]];
                    const auto* c5 = &ternary3_table()[packed[5]];
                    const auto* c6 = &ternary3_table()[packed[6]];
                    const auto* c7 = &ternary3_table()[packed[7]];
                    for (std::size_t j = 0; j < 5; ++j) {
                        sums[0] += input[j] * (*c0)[j]; sums[1] += input[j] * (*c1)[j];
                        sums[2] += input[j] * (*c2)[j]; sums[3] += input[j] * (*c3)[j];
                        sums[4] += input[j] * (*c4)[j]; sums[5] += input[j] * (*c5)[j];
                        sums[6] += input[j] * (*c6)[j]; sums[7] += input[j] * (*c7)[j];
                    }
                }
#endif
        if (complete < stride) {
          for (std::size_t row = 0; row < 8; ++row) {
            const auto &codes = ternary3_table()[bytes[ternary3_offset(
                r + row, complete, stride)]];
            for (std::size_t c = complete * 5; c < in; ++c)
              sums[row] += x[c] * codes[c - complete * 5];
          }
        }
        for (std::size_t row = 0; row < 8; ++row)
          y[r + row] = sums[row] * scales[r + row];
      }
      for (; r < end; ++r) {
        float sum = 0.0f;
        for (std::size_t block = 0; block < complete; ++block) {
          const auto &codes =
              ternary3_table()[bytes[ternary3_offset(r, block, stride)]];
          const float *input = x.data() + block * 5;
          sum += input[0] * codes[0];
          sum += input[1] * codes[1];
          sum += input[2] * codes[2];
          sum += input[3] * codes[3];
          sum += input[4] * codes[4];
        }
        if (complete < stride) {
          const auto &codes =
              ternary3_table()[bytes[ternary3_offset(r, complete, stride)]];
          for (std::size_t c = complete * 5; c < in; ++c)
            sum += x[c] * codes[c - complete * 5];
        }
        y[r] = sum * scales[r];
      }
    });
  } else if (kind == WeightKind::ternary2) {
    const std::size_t stride = in / 4;
    parallel_rows(out, [&](std::size_t begin, std::size_t end) {
      for (std::size_t r = begin; r < end; ++r) {
        float sum = 0.0f;
        for (std::size_t block = 0; block < stride; ++block) {
          unsigned packed = bytes[r * stride + block];
          for (std::size_t j = 0; j < 4; ++j) {
            const unsigned code = (packed >> (2 * j)) & 3;
            if (code == 0)
              sum -= x[block * 4 + j];
            else if (code == 2)
              sum += x[block * 4 + j];
          }
        }
        y[r] = sum * scales[r];
      }
    });
  } else if (kind == WeightKind::rvq) {
    const std::size_t G = in / group, chunks = padded_out / 64;
    const auto *cb = reinterpret_cast<const float *>(bytes.data());
    const std::size_t cb_floats = static_cast<std::size_t>(stages) * group * 16;
    const auto *idx = bytes.data() + cb_floats * sizeof(float);
    // Every row selects one of the same 16 codebook vectors for each input
    // group. Compute those 16 dot products once, then rows only perform
    // byte decode and lookup. This is the essential compressed RVQ kernel.
    thread_local std::vector<float> lookup_storage;
    const std::size_t lookup_size = static_cast<std::size_t>(stages) * G * 16;
    lookup_storage.resize(lookup_size);
    std::span<float> lookup(lookup_storage.data(), lookup_size);
    for (std::size_t stage = 0; stage < stages; ++stage)
      for (std::size_t g = 0; g < G; ++g) {
#if defined(__aarch64__)
        float32x4_t a = vdupq_n_f32(0.0f), b = vdupq_n_f32(0.0f);
        float32x4_t c = vdupq_n_f32(0.0f), d = vdupq_n_f32(0.0f);
        for (std::size_t j = 0; j < group; ++j) {
          const float *codes = cb + (stage * group + j) * 16;
          const float value = x[g * group + j];
          a = vfmaq_n_f32(a, vld1q_f32(codes), value);
          b = vfmaq_n_f32(b, vld1q_f32(codes + 4), value);
          c = vfmaq_n_f32(c, vld1q_f32(codes + 8), value);
          d = vfmaq_n_f32(d, vld1q_f32(codes + 12), value);
        }
        float *destination = lookup.data() + (stage * G + g) * 16;
        vst1q_f32(destination, a);
        vst1q_f32(destination + 4, b);
        vst1q_f32(destination + 8, c);
        vst1q_f32(destination + 12, d);
#else
        for (std::size_t code = 0; code < 16; ++code) {
          float acc = 0.0f;
          for (std::size_t j = 0; j < group; ++j)
            acc += x[g * group + j] * cb[(stage * group + j) * 16 + code];
          lookup[(stage * G + g) * 16 + code] = acc;
        }
#endif
      }
    parallel_rows(out, [&](std::size_t begin, std::size_t end) {
      std::size_t r = begin;
      for (; r + 8 <= end; r += 8) {
        float sums[8]{};
        std::size_t chunks8[8], rows8[8];
        bool high8[8];
        for (std::size_t lane = 0; lane < 8; ++lane) {
          chunks8[lane] = (r + lane) / 64;
          rows8[lane] = (r + lane) % 32;
          high8[lane] = ((r + lane) % 64) >= 32;
        }
        for (std::size_t stage = 0; stage < stages; ++stage)
          for (std::size_t g = 0; g < G; ++g) {
            const float *values = lookup.data() + (stage * G + g) * 16;
            for (std::size_t lane = 0; lane < 8; ++lane) {
              const std::size_t io =
                  ((stage * chunks + chunks8[lane]) * G + g) * 32 + rows8[lane];
              const std::uint8_t packed = idx[io];
              sums[lane] += values[high8[lane] ? packed >> 4 : packed & 15];
            }
          }
        for (std::size_t lane = 0; lane < 8; ++lane)
          y[r + lane] = sums[lane] * scales[r + lane];
      }
      for (; r < end; ++r) {
        const std::size_t chunk = r / 64, half = (r % 64) / 32, row = r % 32;
        float acc = 0.0f;
        for (std::size_t stage = 0; stage < stages; ++stage)
          for (std::size_t g = 0; g < G; ++g) {
            const std::size_t io =
                ((stage * chunks + chunk) * G + g) * 32 + row;
            const std::uint8_t packed = idx[io];
            const std::size_t code = half == 0 ? packed & 15 : packed >> 4;
            acc += lookup[(stage * G + g) * 16 + code];
          }
        y[r] = acc * scales[r];
      }
    });
  } else
    throw std::runtime_error(name + " unsupported matrix kind");
  if (active_profile) {
    const double elapsed = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - started)
                               .count();
    if (kind == WeightKind::rvq) {
      active_profile->rvq += elapsed;
      ++active_profile->rvq_calls;
    } else if (kind == WeightKind::ternary2 || kind == WeightKind::ternary3) {
      active_profile->ternary += elapsed;
      ++active_profile->ternary_calls;
    } else {
      active_profile->dense += elapsed;
      ++active_profile->dense_calls;
    }
  }
}

void Tensor::matvec_pair_into(const Tensor &other, std::span<const float> x,
                              std::span<float> y,
                              std::span<float> other_y) const {
  if (kind != WeightKind::ternary3 || other.kind != WeightKind::ternary3 ||
      in != other.in || out != other.out || x.size() != in || y.size() != out ||
      other_y.size() != out || ternary_reference_enabled() ||
      (dotprod_ffn_group() &&
       ((!ternary_dotprod.empty() && !other.ternary_dotprod.empty()) ||
        (!ternary_dotprod_compact.empty() &&
         !other.ternary_dotprod_compact.empty())))) {
    matvec_into(x, y);
    other.matvec_into(x, other_y);
    return;
  }
  const auto started = active_profile ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
  const std::size_t stride = (in + 4) / 5, complete = in / 5;
  std::fill(y.begin(), y.end(), 0.0f);
  std::fill(other_y.begin(), other_y.end(), 0.0f);
  parallel_rows(out, [&](std::size_t begin, std::size_t end) {
    auto scalar_row = [&](std::size_t row, const Tensor &weight) {
      float sum = 0.0f;
      for (std::size_t block = 0; block < complete; ++block) {
        const auto &codes =
            ternary3_table()[weight.bytes[ternary3_offset(row, block, stride)]];
        const float *input = x.data() + block * 5;
        sum += input[0] * codes[0];
        sum += input[1] * codes[1];
        sum += input[2] * codes[2];
        sum += input[3] * codes[3];
        sum += input[4] * codes[4];
      }
      if (complete < stride) {
        const auto &codes = ternary3_table()[weight.bytes[ternary3_offset(
            row, complete, stride)]];
        for (std::size_t c = complete * 5; c < in; ++c)
          sum += x[c] * codes[c - complete * 5];
      }
      return sum * weight.scales[row];
    };
    std::size_t row = begin;
    for (; row < end && row % 16 != 0; ++row) {
      y[row] = scalar_row(row, *this);
      other_y[row] = scalar_row(row, other);
    }
#if defined(__aarch64__)
    for (; row + 16 <= end; row += 16) {
      const std::size_t base = row;
      float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), a2 = vdupq_n_f32(0),
                  a3 = vdupq_n_f32(0);
      float32x4_t b0 = vdupq_n_f32(0), b1 = vdupq_n_f32(0), b2 = vdupq_n_f32(0),
                  b3 = vdupq_n_f32(0);
#if defined(__clang__)
#pragma clang loop unroll_count(2)
#endif
      for (std::size_t column = 0; column < in; ++column) {
        const int8x16_t ca = unpack_ternary_nibbles(
            ternary_nibbles.data() + ternary_nibble_offset(base, column, in));
        const int8x16_t cb =
            unpack_ternary_nibbles(other.ternary_nibbles.data() +
                                   ternary_nibble_offset(base, column, in));
        const int16x8_t al = vmovl_s8(vget_low_s8(ca)),
                        ah = vmovl_s8(vget_high_s8(ca)),
                        bl = vmovl_s8(vget_low_s8(cb)),
                        bh = vmovl_s8(vget_high_s8(cb));
        a0 = vfmaq_n_f32(a0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(al))),
                         x[column]);
        a1 = vfmaq_n_f32(a1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(al))),
                         x[column]);
        a2 = vfmaq_n_f32(a2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(ah))),
                         x[column]);
        a3 = vfmaq_n_f32(a3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(ah))),
                         x[column]);
        b0 = vfmaq_n_f32(b0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(bl))),
                         x[column]);
        b1 = vfmaq_n_f32(b1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(bl))),
                         x[column]);
        b2 = vfmaq_n_f32(b2, vcvtq_f32_s32(vmovl_s16(vget_low_s16(bh))),
                         x[column]);
        b3 = vfmaq_n_f32(b3, vcvtq_f32_s32(vmovl_s16(vget_high_s16(bh))),
                         x[column]);
      }
      float sa[16], sb[16];
      vst1q_f32(sa, a0);
      vst1q_f32(sa + 4, a1);
      vst1q_f32(sa + 8, a2);
      vst1q_f32(sa + 12, a3);
      vst1q_f32(sb, b0);
      vst1q_f32(sb + 4, b1);
      vst1q_f32(sb + 8, b2);
      vst1q_f32(sb + 12, b3);
      for (std::size_t lane = 0; lane < 16; ++lane) {
        y[row + lane] = sa[lane] * scales[row + lane];
        other_y[row + lane] = sb[lane] * other.scales[row + lane];
      }
    }
#endif
    for (; row + 8 <= end; row += 8) {
#if defined(__aarch64__)
      float32x4_t a_lo = vdupq_n_f32(0.0f), a_hi = vdupq_n_f32(0.0f);
      float32x4_t b_lo = vdupq_n_f32(0.0f), b_hi = vdupq_n_f32(0.0f);
      for (std::size_t block = 0; block < complete; ++block) {
        const float *input = x.data() + block * 5;
        uint16x8_t pa = vmovl_u8(
            vld1_u8(bytes.data() + ternary3_offset(row, block, stride)));
        uint16x8_t pb = vmovl_u8(
            vld1_u8(other.bytes.data() + ternary3_offset(row, block, stride)));
        for (std::size_t j = 0; j < 5; ++j) {
          const uint16x8_t qa = vshrq_n_u16(vmulq_n_u16(pa, 171), 9);
          const uint16x8_t qb = vshrq_n_u16(vmulq_n_u16(pb, 171), 9);
          const int16x8_t da = vsubq_s16(
              vreinterpretq_s16_u16(vsubq_u16(pa, vmulq_n_u16(qa, 3))),
              vdupq_n_s16(1));
          const int16x8_t db = vsubq_s16(
              vreinterpretq_s16_u16(vsubq_u16(pb, vmulq_n_u16(qb, 3))),
              vdupq_n_s16(1));
          a_lo = vfmaq_n_f32(a_lo, vcvtq_f32_s32(vmovl_s16(vget_low_s16(da))),
                             input[j]);
          a_hi = vfmaq_n_f32(a_hi, vcvtq_f32_s32(vmovl_s16(vget_high_s16(da))),
                             input[j]);
          b_lo = vfmaq_n_f32(b_lo, vcvtq_f32_s32(vmovl_s16(vget_low_s16(db))),
                             input[j]);
          b_hi = vfmaq_n_f32(b_hi, vcvtq_f32_s32(vmovl_s16(vget_high_s16(db))),
                             input[j]);
          pa = qa;
          pb = qb;
        }
      }
      float sa[8], sb[8];
      vst1q_f32(sa, a_lo);
      vst1q_f32(sa + 4, a_hi);
      vst1q_f32(sb, b_lo);
      vst1q_f32(sb + 4, b_hi);
      for (std::size_t lane = 0; lane < 8; ++lane) {
        if (complete < stride) {
          const auto &ca = ternary3_table()[bytes[ternary3_offset(
              row + lane, complete, stride)]];
          const auto &cb = ternary3_table()[other.bytes[ternary3_offset(
              row + lane, complete, stride)]];
          for (std::size_t c = complete * 5; c < in; ++c) {
            sa[lane] += x[c] * ca[c - complete * 5];
            sb[lane] += x[c] * cb[c - complete * 5];
          }
        }
        y[row + lane] = sa[lane] * scales[row + lane];
        other_y[row + lane] = sb[lane] * other.scales[row + lane];
      }
#else
            for (std::size_t lane = 0; lane < 8; ++lane) {
                y[row + lane] = scalar_row(row + lane, *this); other_y[row + lane] = scalar_row(row + lane, other);
            }
#endif
    }
    for (; row < end; ++row) {
      y[row] = scalar_row(row, *this);
      other_y[row] = scalar_row(row, other);
    }
  });
  if (active_profile) {
    active_profile->ternary += std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - started)
                                   .count();
    active_profile->ternary_calls += 2;
  }
}

void Tensor::matvec_batch4_into(std::span<const float> x,
                                std::span<float> y) const {
  if (x.size() != 4 * in || y.size() != 4 * out)
    throw std::runtime_error(name + " batch4 matvec shape mismatch");
  if (kind == WeightKind::rvq) {
    const auto started = active_profile ? std::chrono::steady_clock::now()
                                        : std::chrono::steady_clock::time_point{};
    const std::size_t groups = in / group, chunks = padded_out / 64;
    const auto *codebook = reinterpret_cast<const float *>(bytes.data());
    const std::size_t codebook_floats = static_cast<std::size_t>(stages) * group * 16;
    const auto *indices = bytes.data() + codebook_floats * sizeof(float);
    const std::size_t lookup_size = static_cast<std::size_t>(stages) * groups * 16;
    thread_local std::vector<float> batch_lookup_storage;
    batch_lookup_storage.resize(4 * lookup_size);
    float *const lookup = batch_lookup_storage.data();
    for (std::size_t stage = 0; stage < stages; ++stage)
      for (std::size_t g = 0; g < groups; ++g) {
#if defined(__aarch64__)
        float32x4_t sums[4][4];
        for (auto &token_sums : sums)
          for (auto &sum : token_sums) sum = vdupq_n_f32(0);
        for (std::size_t column = 0; column < group; ++column) {
          const float *codes = codebook + (stage * group + column) * 16;
          const float32x4_t weights[4] = {vld1q_f32(codes), vld1q_f32(codes + 4),
                                         vld1q_f32(codes + 8), vld1q_f32(codes + 12)};
          for (std::size_t token = 0; token < 4; ++token)
            for (std::size_t lane = 0; lane < 4; ++lane)
              sums[token][lane] = vfmaq_n_f32(
                  sums[token][lane], weights[lane],
                  x[token * in + g * group + column]);
        }
        for (std::size_t token = 0; token < 4; ++token)
          for (std::size_t lane = 0; lane < 4; ++lane)
            vst1q_f32(lookup + token * lookup_size +
                          (stage * groups + g) * 16 + lane * 4,
                      sums[token][lane]);
#else
        for (std::size_t token = 0; token < 4; ++token)
          for (std::size_t code = 0; code < 16; ++code) {
            float sum = 0;
            for (std::size_t column = 0; column < group; ++column)
              sum += x[token * in + g * group + column] *
                     codebook[(stage * group + column) * 16 + code];
            lookup[token * lookup_size + (stage * groups + g) * 16 + code] = sum;
          }
#endif
      }
    parallel_rows(out, [&](std::size_t begin, std::size_t end) {
      for (std::size_t row = begin; row < end; ++row) {
        float sums[4]{};
        const std::size_t chunk = row / 64, half = (row % 64) / 32, lane = row % 32;
        for (std::size_t stage = 0; stage < stages; ++stage)
          for (std::size_t g = 0; g < groups; ++g) {
            const auto packed = indices[((stage * chunks + chunk) * groups + g) * 32 + lane];
            const std::size_t code = half ? packed >> 4 : packed & 15;
            for (std::size_t token = 0; token < 4; ++token)
              sums[token] += lookup[token * lookup_size +
                                    (stage * groups + g) * 16 + code];
          }
        for (std::size_t token = 0; token < 4; ++token)
          y[token * out + row] = sums[token] * scales[row];
      }
    });
    if (active_profile) {
      active_profile->rvq += std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - started).count();
      active_profile->rvq_calls += 4;
    }
    return;
  }
  if (kind != WeightKind::ternary3 || ternary_reference_enabled()) {
    for (std::size_t token = 0; token < 4; ++token)
      matvec_into(x.subspan(token * in, in), y.subspan(token * out, out));
    return;
  }
  const auto started = active_profile ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
  std::fill(y.begin(), y.end(), 0.0f);
  parallel_rows(out, [&](std::size_t begin, std::size_t end) {
    std::size_t row = begin;
    for (; row < end && row % 16 != 0; ++row) {
      for (std::size_t token = 0; token < 4; ++token) {
        float sum = 0.0f;
        const std::size_t stride = (in + 4) / 5;
        for (std::size_t column = 0; column < in; ++column) {
          const auto &codes = ternary3_table()[
              bytes[ternary3_offset(row, column / 5, stride)]];
          sum += x[token * in + column] * codes[column % 5];
        }
        y[token * out + row] = sum * scales[row];
      }
    }
#if defined(__aarch64__)
    for (; row + 16 <= end; row += 16) {
      float32x4_t sums[4][4];
      for (auto &token_sums : sums)
        for (auto &sum : token_sums) sum = vdupq_n_f32(0);
#if defined(__clang__)
#pragma clang loop unroll_count(2)
#endif
      for (std::size_t column = 0; column < in; ++column) {
        const int8x16_t codes = unpack_ternary_nibbles(
            ternary_nibbles.data() + ternary_nibble_offset(row, column, in));
        const int16x8_t lo = vmovl_s8(vget_low_s8(codes));
        const int16x8_t hi = vmovl_s8(vget_high_s8(codes));
        const float32x4_t weights[4] = {
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))),
            vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),
            vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi)))};
        for (std::size_t token = 0; token < 4; ++token)
          for (std::size_t lane = 0; lane < 4; ++lane)
            sums[token][lane] = vfmaq_n_f32(
                sums[token][lane], weights[lane], x[token * in + column]);
      }
      for (std::size_t token = 0; token < 4; ++token) {
        float raw[16];
        for (std::size_t lane = 0; lane < 4; ++lane)
          vst1q_f32(raw + lane * 4, sums[token][lane]);
        for (std::size_t lane = 0; lane < 16; ++lane)
          y[token * out + row + lane] = raw[lane] * scales[row + lane];
      }
    }
#endif
    for (; row < end; ++row) {
      const std::size_t stride = (in + 4) / 5;
      for (std::size_t token = 0; token < 4; ++token) {
        float sum = 0.0f;
        for (std::size_t column = 0; column < in; ++column) {
          const auto &codes = ternary3_table()[
              bytes[ternary3_offset(row, column / 5, stride)]];
          sum += x[token * in + column] * codes[column % 5];
        }
        y[token * out + row] = sum * scales[row];
      }
    }
  });
  if (active_profile) {
    active_profile->ternary += std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - started)
                                   .count();
    active_profile->ternary_calls += 4;
  }
}

void Tensor::matvec_pair_batch4_into(const Tensor &other,
                                     std::span<const float> x,
                                     std::span<float> y,
                                     std::span<float> other_y) const {
  if (in != other.in || out != other.out || x.size() != 4 * in ||
      y.size() != 4 * out || other_y.size() != 4 * other.out)
    throw std::runtime_error(name + " paired batch4 matvec shape mismatch");
  // Each matrix reuses its decoded weights across all four token states. Keeping
  // the two dispatches separate avoids the paired single-token kernel's larger
  // register footprint while retaining exact per-token accumulation order.
  matvec_batch4_into(x, y);
  other.matvec_batch4_into(x, other_y);
}

#if 0 // rejected scheduling experiments; results live in the benchmark report
/* Rejected K/V shared-dispatch experiment retained only in benchmark records.
void Tensor::rvq_pair_into(const Tensor &other, std::span<const float> x,
                           std::span<float> y, std::span<float> other_y) const {
  if (kind != WeightKind::rvq || other.kind != WeightKind::rvq ||
      in != other.in || out != other.out || group != other.group ||
      stages != other.stages || x.size() != in || y.size() != out ||
      other_y.size() != other.out) {
    matvec_into(x, y); other.matvec_into(x, other_y); return;
  }
  const auto started = active_profile ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
  const std::size_t groups=in/group, lookup_size=static_cast<std::size_t>(stages)*groups*16;
  thread_local std::vector<float> pair_lookup_storage;
  pair_lookup_storage.resize(lookup_size*2);
  float *const pair_lookup = pair_lookup_storage.data();
  const Tensor* weights[2]={this,&other};
  std::span<float> outputs[2]={y,other_y};
  for(std::size_t matrix=0;matrix<2;++matrix){
    const auto* codebook=reinterpret_cast<const float*>(weights[matrix]->bytes.data());
    float* lookup=pair_lookup+matrix*lookup_size;
    for(std::size_t stage=0;stage<stages;++stage)for(std::size_t g=0;g<groups;++g){
#if defined(__aarch64__)
      float32x4_t a=vdupq_n_f32(0),b=vdupq_n_f32(0),c=vdupq_n_f32(0),d=vdupq_n_f32(0);
      for(std::size_t j=0;j<group;++j){const float* codes=codebook+(stage*group+j)*16;const float value=x[g*group+j];a=vfmaq_n_f32(a,vld1q_f32(codes),value);b=vfmaq_n_f32(b,vld1q_f32(codes+4),value);c=vfmaq_n_f32(c,vld1q_f32(codes+8),value);d=vfmaq_n_f32(d,vld1q_f32(codes+12),value);}
      float* dst=lookup+(stage*groups+g)*16;vst1q_f32(dst,a);vst1q_f32(dst+4,b);vst1q_f32(dst+8,c);vst1q_f32(dst+12,d);
#else
      for(std::size_t code=0;code<16;++code){float sum=0;for(std::size_t j=0;j<group;++j)sum+=x[g*group+j]*codebook[(stage*group+j)*16+code];lookup[(stage*groups+g)*16+code]=sum;}
#endif
    }
  }
  parallel_rows(out,[&](std::size_t begin,std::size_t end){
    for(std::size_t matrix=0;matrix<2;++matrix){
      const Tensor& weight=*weights[matrix];auto output=outputs[matrix];
      const std::size_t chunks=weight.padded_out/64,cb_floats=static_cast<std::size_t>(stages)*group*16;
      const auto* indices=weight.bytes.data()+cb_floats*sizeof(float);const float* lookup=pair_lookup+matrix*lookup_size;
      std::size_t row=begin;
      for(;row<end&&row%8;++row){const std::size_t chunk=row/64,half=(row%64)/32,lane=row%32;float sum=0;for(std::size_t stage=0;stage<stages;++stage)for(std::size_t g=0;g<groups;++g){const auto packed=indices[((stage*chunks+chunk)*groups+g)*32+lane];sum+=lookup[(stage*groups+g)*16+(half?packed>>4:packed&15)];}output[row]=sum*weight.scales[row];}
      for(;row+8<=end;row+=8){float sums[8]{};for(std::size_t stage=0;stage<stages;++stage)for(std::size_t g=0;g<groups;++g){const float* values=lookup+(stage*groups+g)*16;for(std::size_t lane=0;lane<8;++lane){const std::size_t current=row+lane,chunk=current/64,packed_row=current%32;const auto packed=indices[((stage*chunks+chunk)*groups+g)*32+packed_row];sums[lane]+=values[(current%64)>=32?packed>>4:packed&15];}}for(std::size_t lane=0;lane<8;++lane)output[row+lane]=sums[lane]*weight.scales[row+lane];}
      for(;row<end;++row){const std::size_t chunk=row/64,half=(row%64)/32,lane=row%32;float sum=0;for(std::size_t stage=0;stage<stages;++stage)for(std::size_t g=0;g<groups;++g){const auto packed=indices[((stage*chunks+chunk)*groups+g)*32+lane];sum+=lookup[(stage*groups+g)*16+(half?packed>>4:packed&15)];}output[row]=sum*weight.scales[row];}
    }
  });
  if(active_profile){active_profile->rvq+=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();active_profile->rvq_calls+=2;}
}
*/

/* Rejected Q/K/V shared-dispatch experiment removed after paired benchmarking. */
void Tensor::matvec_triple_into(const Tensor &second, const Tensor &third,
                                std::span<const float> x, std::span<float> y,
                                std::span<float> second_y,
                                std::span<float> third_y) const {
  const Tensor *weights[3] = {this, &second, &third};
  std::span<float> outputs[3] = {y, second_y, third_y};
  bool compatible = x.size() == in;
  for (std::size_t matrix = 0; matrix < 3; ++matrix)
    compatible = compatible && weights[matrix]->kind == WeightKind::rvq &&
                 weights[matrix]->in == in &&
                 outputs[matrix].size() == weights[matrix]->out;
  if (!compatible) {
    matvec_into(x, y);
    second.matvec_into(x, second_y);
    third.matvec_into(x, third_y);
    return;
  }

  const auto started = active_profile ? std::chrono::steady_clock::now()
                                      : std::chrono::steady_clock::time_point{};
  std::size_t lookup_offsets[3]{}, lookup_total = 0;
  for (std::size_t matrix = 0; matrix < 3; ++matrix) {
    lookup_offsets[matrix] = lookup_total;
    lookup_total += static_cast<std::size_t>(weights[matrix]->stages) *
                    (in / weights[matrix]->group) * 16;
    std::fill(outputs[matrix].begin(), outputs[matrix].end(), 0.0f);
  }
  thread_local std::vector<float> triple_lookup_storage;
  triple_lookup_storage.resize(lookup_total);
  float *const triple_lookup = triple_lookup_storage.data();

  for (std::size_t matrix = 0; matrix < 3; ++matrix) {
    const Tensor &weight = *weights[matrix];
    const std::size_t groups = in / weight.group;
    const auto *codebook = reinterpret_cast<const float *>(weight.bytes.data());
    float *lookup = triple_lookup + lookup_offsets[matrix];
    for (std::size_t stage = 0; stage < weight.stages; ++stage)
      for (std::size_t group_index = 0; group_index < groups; ++group_index) {
#if defined(__aarch64__)
        float32x4_t a = vdupq_n_f32(0), b = vdupq_n_f32(0);
        float32x4_t c = vdupq_n_f32(0), d = vdupq_n_f32(0);
        for (std::size_t column = 0; column < weight.group; ++column) {
          const float *codes =
              codebook + (stage * weight.group + column) * 16;
          const float value = x[group_index * weight.group + column];
          a = vfmaq_n_f32(a, vld1q_f32(codes), value);
          b = vfmaq_n_f32(b, vld1q_f32(codes + 4), value);
          c = vfmaq_n_f32(c, vld1q_f32(codes + 8), value);
          d = vfmaq_n_f32(d, vld1q_f32(codes + 12), value);
        }
        float *destination = lookup + (stage * groups + group_index) * 16;
        vst1q_f32(destination, a); vst1q_f32(destination + 4, b);
        vst1q_f32(destination + 8, c); vst1q_f32(destination + 12, d);
#else
        for (std::size_t code = 0; code < 16; ++code) {
          float sum = 0;
          for (std::size_t column = 0; column < weight.group; ++column)
            sum += x[group_index * weight.group + column] *
                   codebook[(stage * weight.group + column) * 16 + code];
          lookup[(stage * groups + group_index) * 16 + code] = sum;
        }
#endif
      }
  }

  const std::size_t offsets[4] = {0, y.size(), y.size() + second_y.size(),
                                  y.size() + second_y.size() + third_y.size()};
  parallel_rows(offsets[3], [&](std::size_t begin, std::size_t end) {
    for (std::size_t matrix = 0; matrix < 3; ++matrix) {
      const std::size_t local_begin = begin > offsets[matrix]
                                          ? begin - offsets[matrix] : 0;
      const std::size_t local_end =
          std::min(end, offsets[matrix + 1]) > offsets[matrix]
              ? std::min(end, offsets[matrix + 1]) - offsets[matrix] : 0;
      if (local_begin >= local_end) continue;
      const Tensor &weight = *weights[matrix];
      const std::size_t groups = in / weight.group, chunks = weight.padded_out / 64;
      const std::size_t codebook_floats =
          static_cast<std::size_t>(weight.stages) * weight.group * 16;
      const auto *indices = weight.bytes.data() + codebook_floats * sizeof(float);
      const float *lookup = triple_lookup + lookup_offsets[matrix];
      auto output = outputs[matrix];
      std::size_t row = local_begin;
      for (; row < local_end && row % 8; ++row) {
        const std::size_t chunk = row / 64, half = (row % 64) / 32, lane = row % 32;
        float sum = 0;
        for (std::size_t stage = 0; stage < weight.stages; ++stage)
          for (std::size_t group_index = 0; group_index < groups; ++group_index) {
            const auto packed = indices[((stage * chunks + chunk) * groups + group_index) * 32 + lane];
            sum += lookup[(stage * groups + group_index) * 16 +
                          (half ? packed >> 4 : packed & 15)];
          }
        output[row] = sum * weight.scales[row];
      }
      for (; row + 8 <= local_end; row += 8) {
        float sums[8]{};
        for (std::size_t stage = 0; stage < weight.stages; ++stage)
          for (std::size_t group_index = 0; group_index < groups; ++group_index) {
            const float *values = lookup + (stage * groups + group_index) * 16;
            for (std::size_t lane = 0; lane < 8; ++lane) {
              const std::size_t current = row + lane, chunk = current / 64;
              const std::size_t packed_row = current % 32;
              const auto packed = indices[((stage * chunks + chunk) * groups + group_index) * 32 + packed_row];
              sums[lane] += values[(current % 64) >= 32 ? packed >> 4 : packed & 15];
            }
          }
        for (std::size_t lane = 0; lane < 8; ++lane)
          output[row + lane] = sums[lane] * weight.scales[row + lane];
      }
      for (; row < local_end; ++row) {
        const std::size_t chunk = row / 64, half = (row % 64) / 32, lane = row % 32;
        float sum = 0;
        for (std::size_t stage = 0; stage < weight.stages; ++stage)
          for (std::size_t group_index = 0; group_index < groups; ++group_index) {
            const auto packed = indices[((stage * chunks + chunk) * groups + group_index) * 32 + lane];
            sum += lookup[(stage * groups + group_index) * 16 +
                          (half ? packed >> 4 : packed & 15)];
          }
        output[row] = sum * weight.scales[row];
      }
    }
  });
  if (active_profile) {
    active_profile->rvq += std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - started).count();
    active_profile->rvq_calls += 3;
  }
}
#endif


ModelFile::ModelFile(const std::filesystem::path &path) : path_(path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot open model: " + path.string());
  char magic[4];
  in.read(magic, 4);
  if (std::memcmp(magic, "SHDW", 4) != 0)
    throw std::runtime_error("bad SHDW magic");
  version_ = read<std::uint32_t>(in);
  if (version_ != 1 && version_ != 2)
    throw std::runtime_error("unsupported SHDW version");
  const auto count = read<std::uint32_t>(in);
  for (std::uint32_t record = 0; record < count; ++record) {
    const auto name_len = read<std::uint32_t>(in);
    const auto name_bytes = read_bytes(in, name_len);
    Tensor t;
    t.name.assign(reinterpret_cast<const char *>(name_bytes.data()),
                  name_bytes.size());
    t.kind = static_cast<WeightKind>(read<std::uint32_t>(in));
    if (t.kind == WeightKind::dense_f32 || t.kind == WeightKind::dense_f16) {
      const auto nd = read<std::uint32_t>(in);
      t.dims.resize(nd);
      for (auto &d : t.dims)
        d = read<std::uint32_t>(in);
      const std::size_t width = t.kind == WeightKind::dense_f32 ? 4 : 2;
      t.bytes = read_bytes(in, product(t.dims) * width);
      if (t.dims.size() == 2) {
        t.out = t.dims[0];
        t.in = t.dims[1];
      }
    } else if (t.kind == WeightKind::rvq) {
      t.out = read<std::uint32_t>(in);
      t.in = read<std::uint32_t>(in);
      t.group = read<std::uint32_t>(in);
      t.stages = read<std::uint32_t>(in);
      t.padded_out = (t.out + 63) & ~63u;
      const std::size_t cb =
          static_cast<std::size_t>(t.stages) * t.group * 16 * 4;
      const std::size_t idx = static_cast<std::size_t>(t.stages) *
                              (t.padded_out / 64) * (t.in / t.group) * 32;
      t.bytes = read_bytes(in, cb + idx);
      const auto sb =
          read_bytes(in, static_cast<std::size_t>(t.padded_out) * 4);
      t.scales.resize(t.padded_out);
      std::memcpy(t.scales.data(), sb.data(), sb.size());
    } else if (t.kind == WeightKind::ternary2 ||
               t.kind == WeightKind::ternary3) {
      t.out = read<std::uint32_t>(in);
      t.in = read<std::uint32_t>(in);
      const std::size_t stride =
          t.kind == WeightKind::ternary2 ? t.in / 4 : (t.in + 4) / 5;
      t.bytes = read_bytes(in, static_cast<std::size_t>(t.out) * stride);
      if (t.kind == WeightKind::ternary3) {
        if (t.out % 8 != 0)
          throw std::runtime_error("base-3 matrix rows must be divisible by 8");
        auto row_major = std::move(t.bytes);
        t.bytes.resize(row_major.size());
        for (std::size_t row = 0; row < t.out; ++row)
          for (std::size_t block = 0; block < stride; ++block)
            t.bytes[ternary3_offset(row, block, stride)] =
                row_major[row * stride + block];
        if (t.out % 16 != 0)
          throw std::runtime_error(
              "base-3 matrix rows must be divisible by 16");
        t.ternary_nibbles.resize(static_cast<std::size_t>(t.out / 2) * t.in);
        for (std::size_t row = 0; row < t.out; row += 16) {
          for (std::size_t column = 0; column < t.in; ++column) {
            auto *expanded = t.ternary_nibbles.data() +
                             ternary_nibble_offset(row, column, t.in);
            for (std::size_t lane = 0; lane < 16; ++lane) {
              const auto &codes = ternary3_table()[t.bytes[ternary3_offset(
                  row + lane, column / 5, stride)]];
              const auto code =
                  static_cast<std::uint8_t>(codes[column % 5]) & 15;
              if (lane < 8)
                expanded[lane] = code;
              else
                expanded[lane - 8] |= static_cast<std::uint8_t>(code << 4);
            }
          }
        }
        if (dotprod_ffn_group() &&
            (t.name.ends_with(".up") || t.name.ends_with(".gt") ||
             t.name.ends_with(".dn"))) {
          if (t.in % 4 != 0)
            throw std::runtime_error("DotProd FFN input must be divisible by 4");
          if (compact_dotprod_ffn())
            t.ternary_dotprod_compact.resize(
                static_cast<std::size_t>(t.out) * t.in / 4);
          else
            t.ternary_dotprod.resize(static_cast<std::size_t>(t.out) * t.in);
          for (std::size_t row = 0; row < t.out; ++row)
            for (std::size_t column = 0; column < t.in; ++column) {
              const auto value = ternary3_table()[t.bytes[ternary3_offset(
                  row, column / 5, stride)]][column % 5];
              if (compact_dotprod_ffn())
                t.ternary_dotprod_compact[ternary_dotprod_compact_offset(
                    row, column, t.in)] |=
                    static_cast<std::uint8_t>(value + 1) << (2 * (column % 4));
              else
                t.ternary_dotprod[ternary_dotprod_offset(row, column, t.in)] =
                    value;
            }
        }
      }
      const auto sb = read_bytes(in, static_cast<std::size_t>(t.out) * 4);
      t.scales.resize(t.out);
      std::memcpy(t.scales.data(), sb.data(), sb.size());
    } else
      throw std::runtime_error("unknown SHDW record kind in " + t.name);
    tensors_.emplace(t.name, std::move(t));
  }
}

const Tensor &ModelFile::at(const std::string &name) const {
  auto it = tensors_.find(name);
  if (it == tensors_.end())
    throw std::runtime_error("model is missing " + name);
  return it->second;
}
bool ModelFile::contains(const std::string &name) const {
  return tensors_.contains(name);
}

FingerprintTable::FingerprintTable(const std::filesystem::path &path)
    : path_(path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    throw std::runtime_error("cannot open fingerprint table: " + path.string());
  std::array<char, 10> magic{};
  in.read(magic.data(), 6);
  if (std::string_view(magic.data(), 6) != "\x93NUMPY")
    throw std::runtime_error("fingerprint table is not NPY");
  const auto major = read<std::uint8_t>(in);
  (void)read<std::uint8_t>(in);
  std::uint32_t header_len =
      major == 1 ? read<std::uint16_t>(in) : read<std::uint32_t>(in);
  auto header = read_bytes(in, header_len);
  std::string h(reinterpret_cast<char *>(header.data()), header.size());
  if (h.find("'|u1'") == std::string::npos &&
      h.find("'descr': '|u1'") == std::string::npos)
    throw std::runtime_error("fingerprint table must be uint8");
  auto raw = std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in), {});
  if (raw.size() % 64 != 0)
    throw std::runtime_error("fingerprint table row width must be 64 bytes");
  count_ = raw.size() / 64;
  if (count_ % 8 != 0)
    throw std::runtime_error("fingerprint row count must be divisible by 8");
  packed_.resize(raw.size());
  packed_blocked_.resize(raw.size());
  for (std::size_t row = 0; row < count_; ++row)
    for (std::size_t byte = 0; byte < 64; ++byte) {
      packed_[fingerprint_offset(row, byte)] = raw[row * 64 + byte];
      packed_blocked_[(row / 8) * 512 + byte * 8 + row % 8] =
          raw[row * 64 + byte];
    }
}

std::vector<float> FingerprintTable::vector(std::uint32_t token) const {
  std::vector<float> result(FPD);
  vector_into(token, result);
  return result;
}
void FingerprintTable::vector_into(std::uint32_t token,
                                   std::span<float> result) const {
  if (token >= count_)
    throw std::runtime_error("token outside vocabulary");
  if (result.size() != FPD)
    throw std::runtime_error("fingerprint output shape mismatch");
  for (std::size_t bit = 0; bit < FPD; ++bit)
    result[bit] =
        (packed_[fingerprint_offset(token, bit / 8)] >> (7 - bit % 8)) & 1
            ? 1.0f
            : -1.0f;
}
std::vector<float>
FingerprintTable::logits(std::span<const float> projected) const {
  std::vector<float> result(count_);
  logits_into(projected, result);
  return result;
}
void FingerprintTable::logits_into(std::span<const float> projected,
                                   std::span<float> result) const {
  if (result.size() != count_)
    throw std::runtime_error("logit output shape mismatch");
  const float norm = 1.0f / std::sqrt(static_cast<float>(FPD));
  if (fast_logits_enabled()) {
    std::array<float, 64 * 256> contribution{};
    for (std::size_t byte = 0; byte < 64; ++byte) {
      const float *input = projected.data() + byte * 8;
      float base = 0.0f;
      for (std::size_t bit = 0; bit < 8; ++bit)
        base -= input[bit];
      float *values = contribution.data() + byte * 256;
      values[0] = base;
      for (unsigned bit = 0, size = 1; bit < 8; ++bit, size <<= 1) {
        const float delta = 2.0f * input[7 - bit];
        for (unsigned code = 0; code < size; ++code)
          values[code + size] = values[code] + delta;
      }
    }
    parallel_rows(count_, [&](std::size_t begin, std::size_t end) {
      const std::uint8_t *__restrict packed_values = packed_blocked_.data();
      const float *__restrict table_values = contribution.data();
      float *__restrict logits = result.data();
      std::size_t row = begin;
      for (; row + 8 <= end; row += 8) {
        float sums[8]{};
        for (std::size_t byte = 0; byte < 64; ++byte) {
          const std::uint8_t *codes =
              packed_values + (row / 8) * 512 + byte * 8;
          const float *values = table_values + byte * 256;
          sums[0] += values[codes[0]];
          sums[1] += values[codes[1]];
          sums[2] += values[codes[2]];
          sums[3] += values[codes[3]];
          sums[4] += values[codes[4]];
          sums[5] += values[codes[5]];
          sums[6] += values[codes[6]];
          sums[7] += values[codes[7]];
        }
        for (std::size_t lane = 0; lane < 8; ++lane)
          logits[row + lane] = sums[lane] * norm;
      }
      for (; row < end; ++row) {
        float sum = 0.0f;
        for (std::size_t byte = 0; byte < 64; ++byte) {
          sum += table_values[byte * 256 +
                              packed_values[fingerprint_offset(row, byte)]];
        }
        logits[row] = sum * norm;
      }
    });
    return;
  }
  parallel_rows(count_, [&](std::size_t begin, std::size_t end) {
    if (logits_reference_enabled()) {
      for (std::size_t row = begin; row < end; ++row) {
        float sum = 0.0f;
        for (std::size_t bit = 0; bit < FPD; ++bit)
          sum +=
              ((packed_[fingerprint_offset(row, bit / 8)] >> (7 - bit % 8)) & 1
                   ? projected[bit]
                   : -projected[bit]);
        result[row] = sum * norm;
      }
      return;
    }
    std::size_t row = begin;
    for (; row < end && row % 8 != 0; ++row) {
      float sum = 0.0f;
      for (std::size_t bit = 0; bit < FPD; ++bit)
        sum += ((packed_[fingerprint_offset(row, bit / 8)] >> (7 - bit % 8)) & 1
                    ? projected[bit]
                    : -projected[bit]);
      result[row] = sum * norm;
    }
#if defined(__aarch64__)
    for (; row + 16 <= end; row += 16) {
      float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0), b0 = vdupq_n_f32(0),
                  b1 = vdupq_n_f32(0);
      for (std::size_t byte = 0; byte < 64; ++byte) {
        const uint8x16_t packed = vcombine_u8(
            vld1_u8(packed_.data() + fingerprint_offset(row, byte)),
            vld1_u8(packed_.data() + fingerprint_offset(row + 8, byte)));
        const float *input = projected.data() + byte * 8;
#define SHADOW_FP(S, I)                                                        \
  do {                                                                         \
    const int8x16_t bits =                                                     \
        vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(packed, S), vdupq_n_u8(1)));   \
    const int8x16_t signs = vsubq_s8(vaddq_s8(bits, bits), vdupq_n_s8(1));     \
    const int16x8_t lo = vmovl_s8(vget_low_s8(signs)),                         \
                    hi = vmovl_s8(vget_high_s8(signs));                        \
    a0 =                                                                       \
        vfmaq_n_f32(a0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))), input[I]); \
    a1 = vfmaq_n_f32(a1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))),          \
                     input[I]);                                                \
    b0 =                                                                       \
        vfmaq_n_f32(b0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))), input[I]); \
    b1 = vfmaq_n_f32(b1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))),          \
                     input[I]);                                                \
  } while (false)
        SHADOW_FP(7, 0);
        SHADOW_FP(6, 1);
        SHADOW_FP(5, 2);
        SHADOW_FP(4, 3);
        SHADOW_FP(3, 4);
        SHADOW_FP(2, 5);
        SHADOW_FP(1, 6);
#undef SHADOW_FP
        {
          const int8x16_t bits =
              vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(1)));
          const int8x16_t signs = vsubq_s8(vaddq_s8(bits, bits), vdupq_n_s8(1));
          const int16x8_t lo = vmovl_s8(vget_low_s8(signs)),
                          hi = vmovl_s8(vget_high_s8(signs));
          a0 = vfmaq_n_f32(a0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo))),
                           input[7]);
          a1 = vfmaq_n_f32(a1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo))),
                           input[7]);
          b0 = vfmaq_n_f32(b0, vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi))),
                           input[7]);
          b1 = vfmaq_n_f32(b1, vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi))),
                           input[7]);
        }
      }
      float sums[16];
      vst1q_f32(sums, a0);
      vst1q_f32(sums + 4, a1);
      vst1q_f32(sums + 8, b0);
      vst1q_f32(sums + 12, b1);
      for (std::size_t lane = 0; lane < 16; ++lane)
        result[row + lane] = sums[lane] * norm;
    }
#endif
    for (; row + 8 <= end; row += 8) {
#if defined(__aarch64__)
      float32x4_t sums_lo = vdupq_n_f32(0.0f), sums_hi = vdupq_n_f32(0.0f);
      for (std::size_t byte = 0; byte < 64; ++byte) {
        uint8x8_t packed =
            vld1_u8(packed_.data() + fingerprint_offset(row, byte));
        const float *input = projected.data() + byte * 8;
        for (std::size_t bit = 0; bit < 8; ++bit) {
          const uint16x8_t masked =
              vandq_u16(vmovl_u8(packed), vdupq_n_u16(1u << (7 - bit)));
          const uint16x8_t one = vshlq_u16(
              masked, vdupq_n_s16(-static_cast<std::int16_t>(7 - bit)));
          const int16x8_t signed_one = vsubq_s16(
              vreinterpretq_s16_u16(vaddq_u16(one, one)), vdupq_n_s16(1));
          sums_lo = vfmaq_n_f32(
              sums_lo, vcvtq_f32_s32(vmovl_s16(vget_low_s16(signed_one))),
              input[bit]);
          sums_hi = vfmaq_n_f32(
              sums_hi, vcvtq_f32_s32(vmovl_s16(vget_high_s16(signed_one))),
              input[bit]);
        }
      }
      float sums[8];
      vst1q_f32(sums, sums_lo);
      vst1q_f32(sums + 4, sums_hi);
#else
            float sums[8]{};
            for (std::size_t byte = 0; byte < 64; ++byte) {
                const float* input = projected.data() + byte * 8;
                for (std::size_t lane = 0; lane < 8; ++lane) {
                    const auto& signs = fingerprint_sign_table()[packed_[fingerprint_offset(row + lane, byte)]];
                    sums[lane] += input[0] * signs[0]; sums[lane] += input[1] * signs[1];
                    sums[lane] += input[2] * signs[2]; sums[lane] += input[3] * signs[3];
                    sums[lane] += input[4] * signs[4]; sums[lane] += input[5] * signs[5];
                    sums[lane] += input[6] * signs[6]; sums[lane] += input[7] * signs[7];
                }
            }
#endif
      for (std::size_t lane = 0; lane < 8; ++lane)
        result[row + lane] = sums[lane] * norm;
    }
    for (; row < end; ++row) {
      float sum = 0.0f;
      for (std::size_t byte = 0; byte < 64; ++byte) {
        const auto &signs =
            fingerprint_sign_table()[packed_[fingerprint_offset(row, byte)]];
        const float *input = projected.data() + byte * 8;
        sum += input[0] * signs[0];
        sum += input[1] * signs[1];
        sum += input[2] * signs[2];
        sum += input[3] * signs[3];
        sum += input[4] * signs[4];
        sum += input[5] * signs[5];
        sum += input[6] * signs[6];
        sum += input[7] * signs[7];
      }
      result[row] = sum * norm;
    }
  });
}

void FingerprintTable::logits_into(std::span<const float> projected,
                                   std::span<float> result, const Tensor &bias,
                                   std::size_t *argmax) const {
  logits_into(projected, result);
  if (bias.kind != WeightKind::dense_f32 &&
      (bias.kind != WeightKind::dense_f16 ||
       bias.bytes.size() != result.size() * sizeof(std::uint16_t)))
    throw std::runtime_error("unsupported logits bias");
  const auto dense = bias.kind == WeightKind::dense_f32
                         ? bias.dense_f32() : std::span<const float>{};
  const auto *half = bias.kind == WeightKind::dense_f16
      ? reinterpret_cast<const std::uint16_t *>(bias.bytes.data()) : nullptr;
  struct Candidate { float value; std::size_t index; };
  std::array<Candidate, 32> candidates{};
  std::atomic<std::size_t> candidate_count{0};
  parallel_rows(result.size(), [&](std::size_t begin, std::size_t end) {
    float local_peak = -std::numeric_limits<float>::infinity();
    std::size_t local_index = begin;
    for (std::size_t i = begin; i < end; ++i) {
      result[i] += half ? half_to_float(half[i]) : dense[i];
      if (argmax && result[i] > local_peak) { local_peak=result[i]; local_index=i; }
    }
    if (argmax) candidates[candidate_count.fetch_add(1,std::memory_order_relaxed)]={local_peak,local_index};
  });
  if (argmax) {
    Candidate best{-std::numeric_limits<float>::infinity(), 0};
    for (std::size_t i = 0; i < candidate_count.load(std::memory_order_relaxed);
         ++i)
      if (candidates[i].value > best.value ||
          (candidates[i].value == best.value &&
           candidates[i].index < best.index))
        best = candidates[i];
    *argmax = best.index;
  }
}

struct Runtime::Impl {
  struct Scratch {
    std::vector<float> input = std::vector<float>(FPD);
    std::vector<float> x = std::vector<float>(D), z = std::vector<float>(D);
    std::vector<float> q = std::vector<float>(NH * HD),
                       k = std::vector<float>(NKV * HD),
                       v = std::vector<float>(NKV * HD);
    std::vector<float> normalized = std::vector<float>(HD),
                       attended = std::vector<float>(NH * HD);
    std::vector<float> projected = std::vector<float>(D),
                       h = std::vector<float>(D), down = std::vector<float>(D);
    std::vector<float> up = std::vector<float>(4224),
                       gt = std::vector<float>(4224);
    std::vector<float> sq = std::vector<float>(D),
                       recalled = std::vector<float>(D),
                       joined = std::vector<float>(2 * D);
    std::vector<float> hidden = std::vector<float>(4224),
                       structural = std::vector<float>(D),
                       final = std::vector<float>(D);
    std::vector<float> fingerprint = std::vector<float>(FPD),
                       logits = std::vector<float>(131072);
    std::size_t greedy_token = 0;
    std::vector<float> score;
  } scratch;
  std::array<LayerCache, L> cache;
  std::array<LayerWeights, L> weights;
  std::vector<std::array<float, D>> trunk;
  std::uint64_t position = 0;
  const Tensor *embedding, *step_wq, *step_cin, *step_cout, *step_nf, *final_nf,
      *head, *bias;
};

Runtime::Runtime(const std::filesystem::path &model,
                 const std::filesystem::path &table)
    : model_(model), table_(table), impl_(std::make_unique<Impl>()) {
  for (std::size_t layer = 0; layer < L; ++layer) {
    const std::string p = "b." + std::to_string(layer) + ".";
    auto &w = impl_->weights[layer];
    w.n1 = &model_.at(p + "n1.w");
    w.n2 = &model_.at(p + "n2.w");
    w.q = &model_.at(p + "q");
    w.k = &model_.at(p + "k");
    w.v = &model_.at(p + "v");
    w.o = &model_.at(p + "o");
    w.qn = &model_.at(p + "qn.w");
    w.kn = &model_.at(p + "kn.w");
    w.g = &model_.at(p + "g");
    w.alpha = &model_.at(p + "alpha");
    w.up = &model_.at(p + "up");
    w.gt = &model_.at(p + "gt");
    w.dn = &model_.at(p + "dn");
    if (model_.contains(p + "kcodec.sign")) {
      w.k_sign = &model_.at(p + "kcodec.sign");
      w.k_mu = &model_.at(p + "kcodec.mu");
      w.k_ctv = &model_.at(p + "kcodec.ctv");
      w.k_low = &model_.at(p + "kcodec.low");
      w.k_high = &model_.at(p + "kcodec.high");
      w.v_sign = &model_.at(p + "vcodec.sign");
      w.v_mu = &model_.at(p + "vcodec.mu");
      w.v_ctv = &model_.at(p + "vcodec.ctv");
      w.v_low = &model_.at(p + "vcodec.low");
      w.v_high = &model_.at(p + "vcodec.high");
    }
  }
  impl_->embedding = &model_.at("emb.weight");
  impl_->step_wq = &model_.at("step.Wq");
  impl_->step_cin = &model_.at("step.cin");
  impl_->step_cout = &model_.at("step.cout");
  impl_->step_nf = &model_.at("step.nf.w");
  impl_->final_nf = &model_.at("nf.w");
  impl_->head = &model_.at("head.weight");
  impl_->bias = &model_.at("tb");
}
Runtime::~Runtime() = default;

GenerationStats Runtime::generate(std::span<const std::uint32_t> prompt,
                                  const GenerationOptions &options) {
  ProfileCounters profile;
  ProfileScope profile_scope(options.profile ? &profile : nullptr);
  if (!options.archive.empty()) {
    if (model_.version() < 2 || !model_.contains("b.0.kcodec.sign"))
      throw std::runtime_error(
          "cold-KV inference requires a .shdw v2 exported with --with-codecs");
    validate_archive_assets(options.archive, model_.path(), table_.path());
  }
  for (auto &cache : impl_->cache) {
    for (auto &keys : cache.keys) keys.clear();
    for (auto &values : cache.values) values.clear();
  }
  impl_->trunk.clear();
  impl_->position = 0;
  GenerationStats stats;
  stats.load_seconds = load_seconds_;
  std::vector<std::uint32_t> history(prompt.begin(), prompt.end());
  std::vector<std::vector<float>> dumped_logits;
  std::mt19937_64 rng(options.seed);

  auto step = [&](std::uint32_t token, const float *transformer_output = nullptr) {
    auto &s = impl_->scratch;
    auto timed = [&](double &counter, auto &&operation) {
      if (!options.profile) {
        operation();
        return;
      }
      const auto started = std::chrono::steady_clock::now();
      operation();
      counter += std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - started)
                     .count();
    };
    if (transformer_output)
      std::copy_n(transformer_output, D, s.x.begin());
    else {
      table_.vector_into(token, s.input);
      timed(profile.embedding,
            [&] { impl_->embedding->matvec_into(s.input, s.x); });
    }
    auto &x = s.x;
    if (options.trace)
      std::cerr << "TRACE emb " << x[0] << ' ' << x[1] << ' ' << dot(x, x)
                << '\n';
    if (!transformer_output)
      for (std::size_t layer = 0; layer < L; ++layer) {
      const auto &w = impl_->weights[layer];
      rms_into(x, *w.n1, s.z);
      auto &z = s.z;
      timed(profile.qkv, [&] {
        w.q->matvec_into(z, s.q);
        w.k->matvec_into(z, s.k);
        w.v->matvec_into(z, s.v);
      });
      auto &q = s.q;
      auto &k = s.k;
      auto &v = s.v;
      std::array<float, NH * HD> archive_q{};
      for (std::size_t head = 0; head < NH; ++head) {
        auto qh = std::span<float>(q).subspan(head * HD, HD);
        rms_into(qh, *w.qn, s.normalized);
        std::copy(s.normalized.begin(), s.normalized.end(), qh.begin());
        rope_inplace(qh, impl_->position);
        std::copy(qh.begin(), qh.end(), archive_q.begin() + head * HD);
        pot_inplace(qh);
        bfloat16_round_inplace(qh);
      }
      std::array<float, NKV * HD> ka{}, va{};
      for (std::size_t head = 0; head < NKV; ++head) {
        auto kh = std::span<float>(k).subspan(head * HD, HD);
        rms_into(kh, *w.kn, s.normalized);
        std::copy(s.normalized.begin(), s.normalized.end(), kh.begin());
        rope_inplace(kh, impl_->position);
        pot_inplace(kh);
        bfloat16_round_inplace(kh);
        auto vh = std::span<float>(v).subspan(head * HD, HD);
        pot_inplace(vh);
        bfloat16_round_inplace(vh);
        std::copy(kh.begin(), kh.end(), ka.begin() + head * HD);
        std::copy(vh.begin(), vh.end(), va.begin() + head * HD);
      }
      auto &cache = impl_->cache[layer];
      for (std::size_t kvh = 0; kvh < NKV; ++kvh) {
        std::array<float, HD> key{}, value{};
        std::copy_n(ka.data() + kvh * HD, HD, key.data());
        std::copy_n(va.data() + kvh * HD, HD, value.data());
        cache.keys[kvh].push_back(key);
        cache.values[kvh].push_back(value);
        if (cache.keys[kvh].size() > 2048) {
          cache.keys[kvh].erase(cache.keys[kvh].begin());
          cache.values[kvh].erase(cache.values[kvh].begin());
        }
      }
      std::array<std::vector<std::array<float, HD>>, NKV> cold_keys,
          cold_values;
      if (!options.archive.empty())
        for (std::size_t kvh = 0; kvh < NKV; ++kvh) {
          const std::string p = "b." + std::to_string(layer) + ".";
          std::array<float, HD> grouped{};
          for (std::size_t group = 0; group < NH / NKV; ++group)
            for (std::size_t j = 0; j < HD; ++j)
              grouped[j] +=
                  archive_q[(kvh * (NH / NKV) + group) * HD + j] / (NH / NKV);
          const auto code = codec_pack(grouped, model_, p + "kcodec.", kvh);
          const bool use_metal =
              options.archive_backend == "metal" ||
              (options.archive_backend == "auto" && metal_available() &&
               std::filesystem::file_size(options.archive) >=
                   64ull * 1024 * 1024);
          auto shortlist =
              use_metal ? scan_archive_metal(options.archive, code, layer, kvh,
                                             options.archive_top_k)
                        : scan_archive_cpu(options.archive, code, layer, kvh,
                                           options.archive_top_k);
          auto packed =
              gather_archive(options.archive, layer, kvh, shortlist.indices);
          for (std::size_t row = 0; row < shortlist.indices.size(); ++row) {
            const auto begin = row * packed.packed_width;
            cold_keys[kvh].push_back(
                codec_unpack(std::span<const std::uint8_t>(packed.keys)
                                 .subspan(begin, packed.packed_width),
                             model_, p + "kcodec.", kvh));
            cold_values[kvh].push_back(
                codec_unpack(std::span<const std::uint8_t>(packed.values)
                                 .subspan(begin, packed.packed_width),
                             model_, p + "vcodec.", kvh));
          }
        }
      const auto attention_started =
          options.profile ? std::chrono::steady_clock::now()
                          : std::chrono::steady_clock::time_point{};
      std::fill(s.attended.begin(), s.attended.end(), 0.0f);
      auto &attended = s.attended;
      const auto alpha = w.alpha->dense_f32();
      const std::size_t cache_tokens = cache.keys[0].size();
      std::vector<float> shared_scores;
      if (cache_tokens >= 1024 && options.archive.empty()) {
        shared_scores.resize(NH * cache_tokens);
        for (std::size_t kvh = 0; kvh < NKV; ++kvh)
          for (std::size_t t = 0; t < cache_tokens; ++t) {
#if defined(__aarch64__)
            float32x4_t sums[NH / NKV];
            for (auto &sum : sums) sum = vdupq_n_f32(0);
            for (std::size_t j = 0; j < HD; j += 4) {
              const float32x4_t key = vld1q_f32(cache.keys[kvh][t].data() + j);
              for (std::size_t group = 0; group < NH / NKV; ++group) {
                const std::size_t head = kvh * (NH / NKV) + group;
                sums[group] = vfmaq_f32(sums[group],
                    vld1q_f32(q.data() + head * HD + j), key);
              }
            }
            for (std::size_t group = 0; group < NH / NKV; ++group) {
              const std::size_t head = kvh * (NH / NKV) + group;
              const float aq = std::nearbyint(alpha[head] * 4096.0f) / 4096.0f;
              shared_scores[head * cache_tokens + t] =
                  std::floor(aq * bfloat16_round(vaddvq_f32(sums[group])));
            }
#else
            for (std::size_t group = 0; group < NH / NKV; ++group) {
              const std::size_t head = kvh * (NH / NKV) + group;
              const float aq = std::nearbyint(alpha[head] * 4096.0f) / 4096.0f;
              shared_scores[head * cache_tokens + t] = std::floor(
                  aq * bfloat16_round(dot(
                      std::span<const float>(q).subspan(head * HD, HD),
                      cache.keys[kvh][t])));
            }
#endif
          }
      }
      for (std::size_t head = 0; head < NH; ++head) {
        const std::size_t kvh = head / (NH / NKV);
        auto qh = std::span<const float>(q).subspan(head * HD, HD);
        const std::size_t cold_count = cold_keys[kvh].size();
        s.score.assign(cold_count + cache_tokens, 0.0f);
        auto &score = s.score;
        float peak = -std::numeric_limits<float>::infinity();
        for (std::size_t t = 0; t < cold_count; ++t) {
          const float aq = std::nearbyint(alpha[head] * 4096.0f) / 4096.0f;
          score[t] =
              std::floor(aq * bfloat16_round(dot(qh, cold_keys[kvh][t])));
          peak = std::max(peak, score[t]);
        }
        for (std::size_t t = 0; t < cache_tokens; ++t) {
          const float aq = std::nearbyint(alpha[head] * 4096.0f) / 4096.0f;
          const float raw = shared_scores.empty()
              ? aq * bfloat16_round(dot(qh, cache.keys[kvh][t]))
              : shared_scores[head * cache_tokens + t];
          score[cold_count + t] = shared_scores.empty() ? std::floor(raw) : raw;
          peak = std::max(peak, score[cold_count + t]);
          if (options.trace)
            std::cerr << "TRACE score " << impl_->position << ' ' << layer
                      << ' ' << head << ' ' << t << ' ' << raw << ' '
                      << score[cold_count + t] << '\n';
        }
        float denom = 0.0f;
        for (float &s : score) {
          s = std::exp2(std::max(s - peak, -15.0f));
          denom += s;
        }
        for (float &weight : score)
          weight = bfloat16_round(weight / denom);
        for (std::size_t t = 0; t < cold_count; ++t)
          for (std::size_t j = 0; j < HD; ++j)
            attended[head * HD + j] += score[t] * cold_values[kvh][t][j];
        for (std::size_t t = 0; t < cache_tokens; ++t) {
          auto vv = std::span<const float>(cache.values[kvh][t]);
          for (std::size_t j = 0; j < HD; ++j)
            attended[head * HD + j] += score[cold_count + t] * vv[j];
        }
        for (std::size_t j = 0; j < HD; ++j)
          attended[head * HD + j] = bfloat16_round(attended[head * HD + j]);
      }
      if (options.profile)
        profile.attention +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          attention_started)
                .count();
      const auto gate = w.g->dense_f32();
      for (std::size_t i = 0; i < attended.size(); ++i)
        attended[i] *= 1.0f / (1.0f + std::exp(-gate[i]));
      timed(profile.output, [&] { w.o->matvec_into(attended, s.projected); });
      auto &projected = s.projected;
      add_inplace(x, projected);
      rms_into(x, *w.n2, s.h);
      auto &h = s.h;
      dump_ffn_activation("h", layer, h);
      timed(profile.ffn_up_gate,
            [&] { w.up->matvec_pair_into(*w.gt, h, s.up, s.gt); });
      auto &up = s.up;
      auto &gt = s.gt;
      if (options.trace && layer == 0)
        std::cerr << "TRACE l0ffnraw up " << up[0] << ' ' << up[1] << " gt "
                  << gt[0] << ' ' << gt[1] << '\n';
      silu_multiply_inplace(up, gt);
      dump_ffn_activation("up", layer, up);
      timed(profile.ffn_down, [&] { w.dn->matvec_into(up, s.down); });
      auto &down = s.down;
      add_inplace(x, down);
      if (options.trace && layer == 0)
        std::cerr << "TRACE l0parts v " << v[0] << ' ' << v[1] << " o "
                  << projected[0] << ' ' << projected[1] << " h " << h[0] << ' '
                  << h[1] << " up " << up[0] << ' ' << up[1] << " down "
                  << down[0] << ' ' << down[1] << '\n';
      if (options.trace)
        std::cerr << "TRACE layer" << layer << ' ' << x[0] << ' ' << x[1] << ' '
                  << dot(x, x) << '\n';
    }
    std::array<float, D> trunk{};
    std::copy(x.begin(), x.end(), trunk.begin());
    impl_->trunk.push_back(trunk);
    if (impl_->trunk.size() > 2048)
      impl_->trunk.erase(impl_->trunk.begin());
    const auto structural_started =
        options.profile ? std::chrono::steady_clock::now()
                        : std::chrono::steady_clock::time_point{};
    impl_->step_wq->matvec_into(x, s.sq);
    auto &sq = s.sq;
    s.score.assign(impl_->trunk.size(), 0.0f);
    auto &score = s.score;
    float peak = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < score.size(); ++i) {
      score[i] = dot(sq, impl_->trunk[i]) / std::sqrt(static_cast<float>(D));
      peak = std::max(peak, score[i]);
    }
    float denom = 0;
    for (float &s : score) {
      s = std::exp(s - peak);
      denom += s;
    }
    std::fill(s.recalled.begin(), s.recalled.end(), 0.0f);
    auto &recalled = s.recalled;
    for (std::size_t i = 0; i < score.size(); ++i)
      for (std::size_t j = 0; j < D; ++j)
        recalled[j] += score[i] / denom * impl_->trunk[i][j];
    auto &joined = s.joined;
    std::copy(x.begin(), x.end(), joined.begin());
    std::copy(recalled.begin(), recalled.end(), joined.begin() + D);
    impl_->step_cin->matvec_into(joined, s.hidden);
    auto &hidden = s.hidden;
    silu_inplace(hidden);
    impl_->step_cout->matvec_into(hidden, s.structural);
    auto &structural = s.structural;
    add_inplace(x, structural);
    rms_into(x, *impl_->step_nf, s.final);
    x.swap(s.final);
    if (options.profile)
      profile.structural += std::chrono::duration<double>(
                                std::chrono::steady_clock::now() -
                                structural_started)
                                .count();
    if (options.trace)
      std::cerr << "TRACE struct " << x[0] << ' ' << x[1] << ' ' << dot(x, x)
                << '\n';
    rms_into(x, *impl_->final_nf, s.final);
    timed(profile.head,
          [&] { impl_->head->matvec_into(s.final, s.fingerprint); });
    const auto logits_started = options.profile
                                    ? std::chrono::steady_clock::now()
                                    : std::chrono::steady_clock::time_point{};
    table_.logits_into(s.fingerprint, s.logits, *impl_->bias, &s.greedy_token);
    if (options.profile)
      profile.logits += std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - logits_started)
                            .count();
    auto &logits = s.logits;
    if (options.trace) {
      std::vector<std::size_t> order(logits.size());
      std::iota(order.begin(), order.end(), 0);
      std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                        [&](auto a, auto b) { return logits[a] > logits[b]; });
      std::cerr << "TRACE logits";
      for (std::size_t i = 0; i < 5; ++i)
        std::cerr << ' ' << order[i] << ':' << logits[order[i]];
      std::cerr << '\n';
    }
    ++impl_->position;
    return &logits;
  };

  auto transformer_block4 = [&](std::span<const std::uint32_t, 4> tokens,
                                std::span<float> states) {
    std::vector<float> inputs(4 * FPD), normalized(4 * D), q(4 * NH * HD),
        k(4 * NKV * HD), v(4 * NKV * HD), attended(4 * NH * HD),
        projected(4 * D), hidden(4 * D), up(4 * 4224), gate(4 * 4224),
        down(4 * D);
    for (std::size_t token = 0; token < 4; ++token)
      table_.vector_into(tokens[token],
          std::span<float>(inputs).subspan(token * FPD, FPD));
    impl_->embedding->matvec_batch4_into(inputs, states);
    const std::uint64_t block_position = impl_->position;
    for (std::size_t layer = 0; layer < L; ++layer) {
      const auto &w = impl_->weights[layer];
      for (std::size_t token = 0; token < 4; ++token)
        rms_into(std::span<const float>(states).subspan(token * D, D), *w.n1,
                 std::span<float>(normalized).subspan(token * D, D));
      w.q->matvec_batch4_into(normalized, q);
      w.k->matvec_batch4_into(normalized, k);
      w.v->matvec_batch4_into(normalized, v);
      std::fill(attended.begin(), attended.end(), 0.0f);
      auto &cache = impl_->cache[layer];
      const auto alpha = w.alpha->dense_f32();
      const auto output_gate = w.g->dense_f32();
      for (std::size_t token = 0; token < 4; ++token) {
        auto qs = std::span<float>(q).subspan(token * NH * HD, NH * HD);
        auto ks = std::span<float>(k).subspan(token * NKV * HD, NKV * HD);
        auto vs = std::span<float>(v).subspan(token * NKV * HD, NKV * HD);
        for (std::size_t head = 0; head < NH; ++head) {
          auto qh = qs.subspan(head * HD, HD);
          rms_into(qh, *w.qn, impl_->scratch.normalized);
          std::copy(impl_->scratch.normalized.begin(), impl_->scratch.normalized.end(), qh.begin());
          rope_inplace(qh, block_position + token);
          pot_inplace(qh);
          bfloat16_round_inplace(qh);
        }
        std::array<float, NKV * HD> ka{}, va{};
        for (std::size_t head = 0; head < NKV; ++head) {
          auto kh = ks.subspan(head * HD, HD);
          rms_into(kh, *w.kn, impl_->scratch.normalized);
          std::copy(impl_->scratch.normalized.begin(), impl_->scratch.normalized.end(), kh.begin());
          rope_inplace(kh, block_position + token);
          pot_inplace(kh);
          bfloat16_round_inplace(kh);
          auto vh = vs.subspan(head * HD, HD);
          pot_inplace(vh);
          bfloat16_round_inplace(vh);
          std::copy(kh.begin(), kh.end(), ka.begin() + head * HD);
          std::copy(vh.begin(), vh.end(), va.begin() + head * HD);
        }
        for (std::size_t kvh = 0; kvh < NKV; ++kvh) {
          std::array<float, HD> key{}, value{};
          std::copy_n(ka.data() + kvh * HD, HD, key.data());
          std::copy_n(va.data() + kvh * HD, HD, value.data());
          cache.keys[kvh].push_back(key); cache.values[kvh].push_back(value);
          if (cache.keys[kvh].size() > 2048) {
            cache.keys[kvh].erase(cache.keys[kvh].begin());
            cache.values[kvh].erase(cache.values[kvh].begin());
          }
        }
        auto token_attended = std::span<float>(attended).subspan(token * NH * HD, NH * HD);
        for (std::size_t head = 0; head < NH; ++head) {
          const std::size_t kvh = head / (NH / NKV);
          auto qh = qs.subspan(head * HD, HD);
          const std::size_t cache_tokens = cache.keys[kvh].size();
          std::vector<float> scores(cache_tokens);
          float peak = -std::numeric_limits<float>::infinity();
          const float aq = std::nearbyint(alpha[head] * 4096.0f) / 4096.0f;
          for (std::size_t t = 0; t < scores.size(); ++t) {
            scores[t] = std::floor(
                aq * bfloat16_round(dot(qh, cache.keys[kvh][t])));
            peak = std::max(peak, scores[t]);
          }
          float denom = 0;
          for (float &score : scores) {
            score = std::exp2(std::max(score - peak, -15.0f));
            denom += score;
          }
          for (float &score : scores)
            score = bfloat16_round(score / denom);
          for (std::size_t t = 0; t < scores.size(); ++t) {
            auto vv = std::span<const float>(cache.values[kvh][t]);
            for (std::size_t j = 0; j < HD; ++j)
              token_attended[head * HD + j] += scores[t] * vv[j];
          }
          for (std::size_t j = 0; j < HD; ++j)
            token_attended[head * HD + j] =
                bfloat16_round(token_attended[head * HD + j]);
        }
        for (std::size_t i = 0; i < NH * HD; ++i)
          token_attended[i] *=
              1.0f / (1.0f + std::exp(-output_gate[i]));
      }
      w.o->matvec_batch4_into(attended, projected);
      for (std::size_t i = 0; i < states.size(); ++i)
        states[i] += projected[i];
      for (std::size_t token = 0; token < 4; ++token)
        rms_into(std::span<const float>(states).subspan(token * D, D),
                 *w.n2,
                 std::span<float>(hidden).subspan(token * D, D));
      w.up->matvec_pair_batch4_into(*w.gt, hidden, up, gate);
      silu_multiply_inplace(up, gate);
      w.dn->matvec_batch4_into(up, down);
      for (std::size_t i = 0; i < states.size(); ++i)
        states[i] += down[i];
    }
  };

  auto start = std::chrono::steady_clock::now();
  std::vector<float> *logits = nullptr;
  std::size_t prompt_index = 0;
  std::vector<float> block_states(4 * D);
  const char *batch_value = std::getenv("SHADOW_BATCH_PREFILL");
  const bool use_batch_prefill =
      options.archive.empty() && !options.trace &&
      (!batch_value || std::string_view(batch_value) != "0");
  for (; use_batch_prefill && prompt_index + 4 <= prompt.size();
       prompt_index += 4) {
    std::array<std::uint32_t, 4> tokens{
        prompt[prompt_index], prompt[prompt_index + 1],
        prompt[prompt_index + 2], prompt[prompt_index + 3]};
    transformer_block4(tokens, block_states);
    for (std::size_t token = 0; token < 4; ++token)
      logits = step(tokens[token], block_states.data() + token * D);
  }
  for (; prompt_index < prompt.size(); ++prompt_index)
    logits = step(prompt[prompt_index]);
  auto prefill_done = std::chrono::steady_clock::now();
  const ProfileCounters prefill_profile = profile;
  for (std::size_t i = 0; i < options.tokens; ++i) {
    if (!options.dump_logits.empty())
      dumped_logits.push_back(*logits);
    const auto next = static_cast<std::uint32_t>(
        options.temperature <= 0.0f && options.top_k <= 1 &&
                options.repetition_penalty == 1.0f
            ? impl_->scratch.greedy_token
            : sample_token(*logits, options, history, rng));
    stats.tokens.push_back(next);
    history.push_back(next);
    if (next == 1 || next == 9 || i + 1 == options.tokens)
      break;
    logits = step(next);
  }
  auto done = std::chrono::steady_clock::now();
  stats.prefill_seconds =
      std::chrono::duration<double>(prefill_done - start).count();
  stats.decode_seconds =
      std::chrono::duration<double>(done - prefill_done).count();
  if (!options.dump_logits.empty())
    write_npy_logits(options.dump_logits, dumped_logits);
  if (options.profile) {
    const ProfileCounters decode = profile - prefill_profile;
    std::cerr << "PROFILE {\"decode_s\":" << stats.decode_seconds
              << ",\"embedding_s\":" << decode.embedding
              << ",\"qkv_s\":" << decode.qkv
              << ",\"attention_s\":" << decode.attention
              << ",\"output_s\":" << decode.output
              << ",\"ffn_up_gate_s\":" << decode.ffn_up_gate
              << ",\"ffn_down_s\":" << decode.ffn_down
              << ",\"structural_s\":" << decode.structural
              << ",\"head_s\":" << decode.head
              << ",\"logits_s\":" << decode.logits
              << ",\"format_dense_s\":" << decode.dense
              << ",\"format_rvq_s\":" << decode.rvq
              << ",\"format_ternary_s\":" << decode.ternary
              << ",\"decode_steps\":"
              << (stats.tokens.empty() ? 0 : stats.tokens.size() - 1)
              << "}\n";
  }
  return stats;
}

std::vector<std::uint32_t> parse_token_list(const std::string &text) {
  std::vector<std::uint32_t> out;
  std::size_t pos = 0;
  while (pos < text.size()) {
    while (pos < text.size() &&
           std::isspace(static_cast<unsigned char>(text[pos])))
      ++pos;
    if (pos == text.size())
      break;
    std::size_t end = pos;
    while (end < text.size() &&
           !std::isspace(static_cast<unsigned char>(text[end])))
      ++end;
    std::size_t used = 0;
    auto value = std::stoul(text.substr(pos, end - pos), &used);
    if (used != end - pos)
      throw std::runtime_error("invalid token list");
    out.push_back(static_cast<std::uint32_t>(value));
    pos = end;
  }
  return out;
}

std::uint32_t popcount_xor(std::span<const std::uint8_t> a,
                           std::span<const std::uint8_t> b) {
  if (a.size() != b.size())
    throw std::runtime_error("popcount shape mismatch");
#if defined(__aarch64__)
  uint64x2_t total = vdupq_n_u64(0);
  std::size_t i = 0;
  for (; i + 16 <= a.size(); i += 16) {
    const uint8x16_t counts =
        vcntq_u8(veorq_u8(vld1q_u8(a.data() + i), vld1q_u8(b.data() + i)));
    total = vpadalq_u32(total, vpaddlq_u16(vpaddlq_u8(counts)));
  }
  std::uint32_t result = static_cast<std::uint32_t>(vaddvq_u64(total));
  for (; i < a.size(); ++i)
    result += std::popcount(static_cast<unsigned>(a[i] ^ b[i]));
  return result;
#else
  std::uint32_t result = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    result += std::popcount(static_cast<unsigned>(a[i] ^ b[i]));
  return result;
#endif
}

} // namespace shadowrt
