#include "shadow/model.hpp"
#include "shadow/archive.hpp"
#include <array>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>

int main() {
    const auto ids = shadowrt::parse_token_list("2 8 42 9");
    assert((ids == std::vector<std::uint32_t>{2, 8, 42, 9}));
    std::array<std::uint8_t, 64> a{}, b{}; b[0] = 0xff; b[63] = 3;
    assert(shadowrt::popcount_xor(a, b) == 10);

    // Runtime batch-4 API must exactly match four ordinary RVQ calls.
    shadowrt::Tensor rvq; rvq.name="test.rvq"; rvq.kind=shadowrt::WeightKind::rvq;
    rvq.in=4; rvq.out=64; rvq.padded_out=64; rvq.group=4; rvq.stages=1;
    rvq.bytes.resize(4*16*sizeof(float)+32); rvq.scales.assign(64,1.0f);
    auto* codebook=reinterpret_cast<float*>(rvq.bytes.data());
    for(std::size_t i=0;i<4*16;++i)codebook[i]=static_cast<float>(static_cast<int>(i%9)-4)/8.0f;
    for(std::size_t i=4*16*sizeof(float);i<rvq.bytes.size();++i)rvq.bytes[i]=static_cast<std::uint8_t>(i*13);
    std::array<float,16> batch_input{};for(std::size_t i=0;i<batch_input.size();++i)batch_input[i]=static_cast<float>(i-7)/5.0f;
    std::array<float,256> batch_output{},sequential_output{};
    rvq.matvec_batch4_into(batch_input,batch_output);
    for(std::size_t token=0;token<4;++token)rvq.matvec_into(
        std::span<const float>(batch_input).subspan(token*4,4),
        std::span<float>(sequential_output).subspan(token*64,64));
    assert(std::memcmp(batch_output.data(),sequential_output.data(),sizeof(batch_output))==0);

    const auto archive = std::filesystem::temp_directory_path() / "shadow-native-test.shkv";
    shadowrt::ArchiveHeader header{};
    std::memcpy(header.magic, "SHARKV1", 7);
    header.version = 1; header.header_bytes = sizeof(header); header.layers = 1;
    header.heads = 1; header.packed_width = 2; header.token_count = 4;
    header.keys_offset = sizeof(header); header.values_offset = header.keys_offset + 8;
    header.tokens_offset = header.values_offset + 8;
    const std::array<std::uint8_t, 8> keys{0x00,0x00, 0xff,0xff, 0x01,0x00, 0x00,0x01};
    const std::array<std::uint8_t, 8> values{0,1,2,3,4,5,6,7};
    {
        std::ofstream out(archive, std::ios::binary);
        out.write(reinterpret_cast<const char*>(&header), sizeof(header));
        out.write(reinterpret_cast<const char*>(keys.data()), keys.size());
        out.write(reinterpret_cast<const char*>(values.data()), values.size());
    }
    const std::array<std::uint8_t, 2> query{};
    const auto scan = shadowrt::scan_archive_cpu(archive, query, 0, 0, 3);
    assert((scan.indices == std::vector<std::uint64_t>{0,2,3}));
    assert((scan.distances == std::vector<std::uint32_t>{0,1,1}));
    const auto gathered = shadowrt::gather_archive(archive, 0, 0, scan.indices);
    assert((gathered.values == std::vector<std::uint8_t>{0,1,4,5,6,7}));
    std::filesystem::remove(archive);

    const auto empty_model = std::filesystem::temp_directory_path() / "shadow-empty-model";
    const auto empty_table = std::filesystem::temp_directory_path() / "shadow-empty-table";
    { std::ofstream(empty_model, std::ios::binary); std::ofstream(empty_table, std::ios::binary); }
    constexpr std::array<std::uint8_t, 32> empty_sha{
        0xe3,0xb0,0xc4,0x42,0x98,0xfc,0x1c,0x14,0x9a,0xfb,0xf4,0xc8,0x99,0x6f,0xb9,0x24,
        0x27,0xae,0x41,0xe4,0x64,0x9b,0x93,0x4c,0xa4,0x95,0x99,0x1b,0x78,0x52,0xb8,0x55};
    header.token_count = 0;
    std::copy(empty_sha.begin(), empty_sha.end(), header.model_sha256);
    std::copy(empty_sha.begin(), empty_sha.end(), header.table_sha256);
    { std::ofstream out(archive, std::ios::binary); out.write(reinterpret_cast<const char*>(&header), sizeof(header)); }
    shadowrt::validate_archive_assets(archive, empty_model, empty_table);
    std::filesystem::remove(archive); std::filesystem::remove(empty_model); std::filesystem::remove(empty_table);
    std::cout << "native tests passed\n";
}
