#include "shadow/archive.hpp"
#include "shadow/model.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <queue>
#include <stdexcept>

namespace shadowrt {

ScanResult scan_archive_cpu(const std::filesystem::path& path, std::span<const std::uint8_t> query,
                            std::uint32_t layer, std::uint32_t head, std::size_t top_k) {
    const auto started = std::chrono::steady_clock::now();
    std::ifstream in(path, std::ios::binary); ArchiveHeader header{}; in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || std::memcmp(header.magic, "SHARKV1", 7) || header.version != 1 || header.header_bytes != sizeof(header))
        throw std::runtime_error("invalid SHADOW KV archive");
    if (layer >= header.layers || head >= header.heads || query.size() != header.packed_width) throw std::runtime_error("archive scan shape mismatch");
    using Candidate = std::pair<std::uint32_t, std::uint64_t>; std::priority_queue<Candidate> best;
    const std::uint64_t base = header.keys_offset + ((static_cast<std::uint64_t>(layer) * header.heads + head) * header.token_count * header.packed_width);
    in.seekg(static_cast<std::streamoff>(base)); std::vector<std::uint8_t> key(header.packed_width);
    for (std::uint64_t i = 0; i < header.token_count; ++i) {
        in.read(reinterpret_cast<char*>(key.data()), key.size()); if (!in) throw std::runtime_error("truncated archive keys");
        Candidate c{popcount_xor(query, key), i};
        if (best.size() < top_k) best.push(c); else if (c < best.top()) { best.pop(); best.push(c); }
    }
    ScanResult result; result.backend = "cpu"; result.indices.resize(best.size()); result.distances.resize(best.size());
    for (std::size_t i = best.size(); i-- > 0;) { result.distances[i] = best.top().first; result.indices[i] = best.top().second; best.pop(); }
    result.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(); return result;
}

bool metal_available() { return false; }
ScanResult scan_archive_metal(const std::filesystem::path& path, std::span<const std::uint8_t> query,
                              std::uint32_t layer, std::uint32_t head, std::size_t top_k) {
    auto result = scan_archive_cpu(path, query, layer, head, top_k); result.backend = "cpu-fallback"; return result;
}

} // namespace shadowrt
