#include "shadow/archive.hpp"

#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <stdexcept>

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
    constexpr std::array<std::uint32_t, 64> k = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::array<std::uint32_t, 8> h = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                                      0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    auto compress = [&](const std::uint8_t* block) {
        std::array<std::uint32_t, 64> w{};
        for (std::size_t i=0;i<16;++i) w[i]=(std::uint32_t(block[4*i])<<24)|(std::uint32_t(block[4*i+1])<<16)|(std::uint32_t(block[4*i+2])<<8)|block[4*i+3];
        for (std::size_t i=16;i<64;++i) {
            const auto s0=std::rotr(w[i-15],7)^std::rotr(w[i-15],18)^(w[i-15]>>3);
            const auto s1=std::rotr(w[i-2],17)^std::rotr(w[i-2],19)^(w[i-2]>>10);
            w[i]=w[i-16]+s0+w[i-7]+s1;
        }
        auto a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (std::size_t i=0;i<64;++i) {
            const auto s1=std::rotr(e,6)^std::rotr(e,11)^std::rotr(e,25);
            const auto t1=hh+s1+((e&f)^(~e&g))+k[i]+w[i];
            const auto s0=std::rotr(a,2)^std::rotr(a,13)^std::rotr(a,22);
            const auto t2=s0+((a&b)^(a&c)^(b&c));
            hh=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    };
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot hash " + path.string());
    std::array<std::uint8_t, 64> block{}; std::uint64_t bytes=0;
    while (in.read(reinterpret_cast<char*>(block.data()), block.size())) { compress(block.data()); bytes+=64; }
    const auto tail=static_cast<std::size_t>(in.gcount()); bytes+=tail; block[tail]=0x80;
    if (tail>=56) { std::fill(block.begin()+tail+1,block.end(),0); compress(block.data()); block.fill(0); }
    else std::fill(block.begin()+tail+1,block.end(),0);
    const std::uint64_t bits=bytes*8; for (int i=0;i<8;++i) block[63-i]=static_cast<std::uint8_t>(bits>>(8*i));
    compress(block.data()); std::array<std::uint8_t,32> result{};
    for (std::size_t i=0;i<8;++i) for (int j=0;j<4;++j) result[4*i+j]=static_cast<std::uint8_t>(h[i]>>(24-8*j));
    return result;
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
