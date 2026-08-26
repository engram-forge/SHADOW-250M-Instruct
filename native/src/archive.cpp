#include "shadow/archive.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <stdexcept>

#if defined(__APPLE__)
#include <CommonCrypto/CommonDigest.h>
#endif

namespace shadowrt {
namespace {

ArchiveHeader header_of(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary); ArchiveHeader h{};
    in.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!in || std::memcmp(h.magic, "SHARKV1", 7) || h.version != 1 || h.header_bytes != sizeof(h))
        throw std::runtime_error("invalid SHADOW KV archive");
    return h;
}

std::array<std::uint8_t, 32> sha256(const std::filesystem::path& path) {
#if defined(__APPLE__)
    std::ifstream in(path, std::ios::binary); if (!in) throw std::runtime_error("cannot hash " + path.string());
    CC_SHA256_CTX context; CC_SHA256_Init(&context); std::array<char, 1 << 20> block{};
    while (in) { in.read(block.data(), block.size()); if (in.gcount()) CC_SHA256_Update(&context, block.data(), static_cast<CC_LONG>(in.gcount())); }
    std::array<std::uint8_t, 32> result{}; CC_SHA256_Final(result.data(), &context); return result;
#else
    (void)path; throw std::runtime_error("archive asset hashing is only implemented by the macOS runner");
#endif
}

} // namespace

ArchiveVectors gather_archive(const std::filesystem::path& path, std::uint32_t layer,
                              std::uint32_t head, std::span<const std::uint64_t> indices) {
    const auto h = header_of(path);
    if (layer >= h.layers || head >= h.heads) throw std::runtime_error("archive gather shape mismatch");
    ArchiveVectors result; result.packed_width = h.packed_width;
    result.keys.resize(indices.size() * h.packed_width); result.values.resize(indices.size() * h.packed_width);
    std::ifstream in(path, std::ios::binary);
    for (std::size_t row = 0; row < indices.size(); ++row) {
        if (indices[row] >= h.token_count) throw std::runtime_error("archive result index outside payload");
        for (int kind = 0; kind < 2; ++kind) {
            const auto section = kind == 0 ? h.keys_offset : h.values_offset;
            const std::uint64_t offset = section + ((static_cast<std::uint64_t>(layer) * h.heads + head) * h.token_count + indices[row]) * h.packed_width;
            auto& output = kind == 0 ? result.keys : result.values;
            in.seekg(static_cast<std::streamoff>(offset));
            in.read(reinterpret_cast<char*>(output.data() + row * h.packed_width), h.packed_width);
            if (!in) throw std::runtime_error("truncated archive vector payload");
        }
    }
    return result;
}

void validate_archive_assets(const std::filesystem::path& archive, const std::filesystem::path& model,
                             const std::filesystem::path& table) {
    const auto h = header_of(archive);
    const auto mh = sha256(model);
    const auto th = sha256(table);
    if (!std::equal(mh.begin(), mh.end(), h.model_sha256)) throw std::runtime_error("archive model SHA-256 mismatch");
    if (!std::equal(th.begin(), th.end(), h.table_sha256)) throw std::runtime_error("archive fingerprint table SHA-256 mismatch");
}

} // namespace shadowrt
