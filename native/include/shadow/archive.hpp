#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace shadowrt {

struct ArchiveHeader {
    char magic[8];                 // SHARKV1\0
    std::uint32_t version;
    std::uint32_t header_bytes;
    std::uint32_t layers;
    std::uint32_t heads;
    std::uint32_t packed_width;
    std::uint32_t page_size;
    std::uint64_t token_count;
    std::uint64_t positions_offset;
    std::uint64_t keys_offset;
    std::uint64_t values_offset;
    std::uint64_t tokens_offset;
    std::uint8_t model_sha256[32];
    std::uint8_t table_sha256[32];
    std::uint8_t reserved[120];
};
static_assert(sizeof(ArchiveHeader) == 256);

struct ScanResult {
    std::vector<std::uint64_t> indices;
    std::vector<std::uint32_t> distances;
    std::string backend;
    double milliseconds = 0.0;
};

struct ArchiveVectors {
    std::uint32_t packed_width = 0;
    std::vector<std::uint8_t> keys;
    std::vector<std::uint8_t> values;
};

ScanResult scan_archive_cpu(const std::filesystem::path& path,
                            std::span<const std::uint8_t> query,
                            std::uint32_t layer, std::uint32_t head,
                            std::size_t top_k);
ScanResult scan_archive_metal(const std::filesystem::path& path,
                              std::span<const std::uint8_t> query,
                              std::uint32_t layer, std::uint32_t head,
                              std::size_t top_k);
bool metal_available();
ArchiveVectors gather_archive(const std::filesystem::path& path, std::uint32_t layer,
                              std::uint32_t head, std::span<const std::uint64_t> indices);
void validate_archive_assets(const std::filesystem::path& archive,
                             const std::filesystem::path& model,
                             const std::filesystem::path& table);

} // namespace shadowrt
