#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace shadowrt {

enum class WeightKind : std::uint32_t {
  dense_f32 = 0,
  rvq = 1,
  ternary2 = 3,
  ternary3 = 4,
  dense_f16 = 5
};

struct Tensor {
  std::string name;
  WeightKind kind{};
  std::vector<std::uint32_t> dims;
  std::uint32_t out = 0, in = 0, group = 0, stages = 0, padded_out = 0;
  std::vector<std::uint8_t> bytes;
  std::vector<float> scales;
  // Decode-optimized 4-bit expansion of base-3 weights, interleaved by 16 rows.
  std::vector<std::uint8_t> ternary_nibbles;

  std::vector<float> matvec(std::span<const float> x) const;
  void matvec_into(std::span<const float> x, std::span<float> y) const;
  void matvec_pair_into(const Tensor &other, std::span<const float> x,
                        std::span<float> y, std::span<float> other_y) const;
  void matvec_batch4_into(std::span<const float> x,
                          std::span<float> y) const;
  std::span<const float> dense_f32() const;
};

class ModelFile {
public:
  explicit ModelFile(const std::filesystem::path &path);
  std::uint32_t version() const { return version_; }
  const Tensor &at(const std::string &name) const;
  bool contains(const std::string &name) const;
  const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
  std::uint32_t version_ = 0;
  std::unordered_map<std::string, Tensor> tensors_;
};

class FingerprintTable {
public:
  explicit FingerprintTable(const std::filesystem::path &path);
  std::vector<float> vector(std::uint32_t token) const;
  void vector_into(std::uint32_t token, std::span<float> output) const;
  std::vector<float> logits(std::span<const float> projected) const;
  void logits_into(std::span<const float> projected,
                   std::span<float> output) const;
  void logits_into(std::span<const float> projected, std::span<float> output,
                   const Tensor &bias, std::size_t *argmax = nullptr) const;
  std::size_t size() const { return count_; }
  const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
  std::size_t count_ = 0;
  std::vector<std::uint8_t> packed_;
  std::vector<std::uint8_t> packed_blocked_;
};

struct GenerationOptions {
  std::size_t tokens = 140;
  float temperature = 0.0f;
  std::size_t top_k = 1;
  float repetition_penalty = 1.0f;
  std::uint64_t seed = 0;
  bool stream = false;
  bool trace = false;
  std::filesystem::path archive;
  std::string archive_backend = "auto";
  std::size_t archive_top_k = 32;
  std::filesystem::path dump_logits;
  bool profile = false;
};

struct GenerationStats {
  std::vector<std::uint32_t> tokens;
  double load_seconds = 0.0;
  double prefill_seconds = 0.0;
  double decode_seconds = 0.0;
};

class Runtime {
public:
  Runtime(const std::filesystem::path &model,
          const std::filesystem::path &table);
  ~Runtime();
  GenerationStats generate(std::span<const std::uint32_t> prompt,
                           const GenerationOptions &options);
  std::uint32_t model_version() const { return model_.version(); }

private:
  struct Impl;
  ModelFile model_;
  FingerprintTable table_;
  std::unique_ptr<Impl> impl_;
  double load_seconds_ = 0.0;
};

std::vector<std::uint32_t> parse_token_list(const std::string &text);
std::uint32_t popcount_xor(std::span<const std::uint8_t> a,
                           std::span<const std::uint8_t> b);

} // namespace shadowrt
