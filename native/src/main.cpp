#include "shadow/archive.hpp"
#include "shadow/model.hpp"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void usage(const char* name) {
    std::cerr << "usage: " << name << " model.shdw fp.npy \"t1 t2\" [ngen] "
              << "[--bench] [--stream] [--status] [--temp T] [--topk K] [--rep R] [--seed S]\n"
              << "       generation options: [--archive file.shkv] [--archive-backend auto|cpu|metal] [--archive-topk K]\n"
              << "       " << name << " --scan archive.shkv query_hex layer head k [--backend auto|cpu|metal]\n";
}

std::vector<std::uint8_t> hex_bytes(const std::string& value) {
    if (value.size() % 2) throw std::runtime_error("hex query must contain complete bytes");
    std::vector<std::uint8_t> out(value.size() / 2);
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = static_cast<std::uint8_t>(std::stoul(value.substr(2 * i, 2), nullptr, 16));
    return out;
}

} // namespace

int main(int argc, char** argv) try {
    if (argc > 1 && std::string(argv[1]) == "--capabilities") {
#if defined(__aarch64__)
        constexpr const char* architecture = "arm64";
#else
        constexpr const char* architecture = "unknown";
#endif
        std::cout << "{\"architecture\":\"" << architecture
                  << "\",\"cpu_backend\":\""
#if defined(__aarch64__)
                  << "neon-armv8"
#else
                  << "scalar"
#endif
                  << "\",\"archive_backends\":[";
        const auto backends = shadowrt::available_archive_backends();
        for (std::size_t i = 0; i < backends.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << '"' << backends[i] << '"';
        }
        std::cout << "],\"metal_available\":"
                  << (shadowrt::metal_available() ? "true" : "false")
                  << ",\"archive_auto_threshold_bytes\":67108864}\n";
        return 0;
    }
    if (argc > 1 && std::string(argv[1]) == "--scan") {
        if (argc < 7) { usage(argv[0]); return 2; }
        const auto query = hex_bytes(argv[3]); const auto layer = std::stoul(argv[4]);
        const auto head = std::stoul(argv[5]); const auto k = std::stoul(argv[6]);
        std::string backend = "auto";
        std::size_t repeat = 1;
        for (int i = 7; i + 1 < argc; ++i) {
            if (std::string(argv[i]) == "--backend") backend = argv[++i];
            else if (std::string(argv[i]) == "--repeat") repeat = std::max<std::size_t>(1, std::stoul(argv[++i]));
        }
        if (backend != "auto" && backend != "cpu" && backend != "metal")
            throw std::runtime_error("unsupported archive backend: " + backend);
        if (backend == "metal" && !shadowrt::metal_available())
            throw std::runtime_error("Metal archive backend is unavailable on this platform");
        shadowrt::ScanResult result; std::vector<double> timings; timings.reserve(repeat);
        const bool auto_metal = backend == "auto" && shadowrt::metal_available()
            && std::filesystem::file_size(argv[2]) >= 64ull * 1024 * 1024;
        for (std::size_t run = 0; run < repeat; ++run) {
            if (backend == "metal" || auto_metal)
                result = shadowrt::scan_archive_metal(argv[2], query, layer, head, k);
            else result = shadowrt::scan_archive_cpu(argv[2], query, layer, head, k);
            timings.push_back(result.milliseconds);
        }
        std::sort(timings.begin(), timings.end()); result.milliseconds = timings[timings.size() / 2];
        std::cout << "{\"backend\":\"" << result.backend << "\",\"milliseconds\":"
                  << result.milliseconds << ",\"results\":[";
        for (std::size_t i = 0; i < result.indices.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << "[" << result.indices[i] << ',' << result.distances[i] << ']';
        }
        std::cout << "]}\n"; return 0;
    }
    if (argc < 4) { usage(argv[0]); return 2; }
    shadowrt::GenerationOptions options;
    if (argc > 4 && argv[4][0] != '-') options.tokens = std::stoul(argv[4]);
    bool status = false, bench = false;
    for (int i = 5; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--temp" && i + 1 < argc) options.temperature = std::stof(argv[++i]);
        else if (arg == "--topk" && i + 1 < argc) options.top_k = std::stoul(argv[++i]);
        else if (arg == "--rep" && i + 1 < argc) options.repetition_penalty = std::stof(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) options.seed = std::stoull(argv[++i]);
        else if (arg == "--stream") options.stream = true;
        else if (arg == "--status") status = true;
        else if (arg == "--bench") bench = true;
        else if (arg == "--trace") options.trace = true;
        else if (arg == "--archive" && i + 1 < argc) options.archive = argv[++i];
        else if (arg == "--archive-backend" && i + 1 < argc) options.archive_backend = argv[++i];
        else if (arg == "--archive-topk" && i + 1 < argc) options.archive_top_k = std::stoul(argv[++i]);
        else if (arg == "--dump-logits" && i + 1 < argc) options.dump_logits = argv[++i];
        else if (arg == "--profile") options.profile = true;
        else throw std::runtime_error("unknown or incomplete option: " + arg);
    }
    if (options.archive_backend != "auto" && options.archive_backend != "cpu" &&
        options.archive_backend != "metal")
        throw std::runtime_error("unsupported archive backend: " + options.archive_backend);
    if (options.archive_backend == "metal" && !shadowrt::metal_available())
        throw std::runtime_error("Metal archive backend is unavailable on this platform");
    shadowrt::Runtime runtime(argv[1], argv[2]);
    auto stats = runtime.generate(shadowrt::parse_token_list(argv[3]), options);
    for (auto token : stats.tokens) std::cout << token << ' ';
    std::cout << '\n';
    if (status || bench) {
        const auto decode_steps = stats.tokens.empty() ? 0 : stats.tokens.size() - 1;
        const double speed = stats.decode_seconds > 0 ? decode_steps / stats.decode_seconds : 0;
        std::cerr << "SHADOW arm64 runtime | SHDW v" << runtime.model_version()
                  << " | prefill " << std::fixed << std::setprecision(3) << stats.prefill_seconds << "s"
                  << " | decode " << (decode_steps ? std::to_string(speed) + " tok/s" : "n/a (no decode step)") << '\n';
    }
    return 0;
} catch (const std::exception& error) {
    std::cerr << "shadow: " << error.what() << '\n'; return 1;
}
