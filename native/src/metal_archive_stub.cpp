#include "shadow/archive.hpp"
#include "shadow/model.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <queue>
#include <stdexcept>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace shadowrt {

ScanResult scan_archive_cpu(const std::filesystem::path& path, std::span<const std::uint8_t> query,
                            std::uint32_t layer, std::uint32_t head, std::size_t top_k) {
    const auto started = std::chrono::steady_clock::now();
    const int fd=open(path.c_str(),O_RDONLY); if(fd<0) throw std::runtime_error("cannot open archive");
    struct stat st{}; if(fstat(fd,&st)!=0){close(fd);throw std::runtime_error("cannot stat archive");}
    if(st.st_size<static_cast<off_t>(sizeof(ArchiveHeader))){close(fd);throw std::runtime_error("invalid SHADOW KV archive");}
    void* mapping=mmap(nullptr,static_cast<std::size_t>(st.st_size),PROT_READ,MAP_PRIVATE,fd,0); close(fd);
    if(mapping==MAP_FAILED) throw std::runtime_error("cannot mmap archive");
    struct Mapping { void* p; std::size_t n; ~Mapping(){munmap(p,n);} } guard{mapping,static_cast<std::size_t>(st.st_size)};
    ArchiveHeader header{}; std::memcpy(&header,mapping,sizeof(header));
    if (std::memcmp(header.magic, "SHARKV1", 7) || header.version != 1 || header.header_bytes != sizeof(header)) throw std::runtime_error("invalid SHADOW KV archive");
    if (header.token_count==0 || top_k==0 || top_k>header.token_count || layer >= header.layers || head >= header.heads || query.size() != header.packed_width) throw std::runtime_error("archive scan shape mismatch");
    const auto plane=static_cast<std::uint64_t>(layer)*header.heads+head;
    if (header.packed_width != 0 && header.token_count > UINT64_MAX / header.packed_width)
        throw std::runtime_error("archive key payload is too large");
    const std::uint64_t payload=header.token_count*header.packed_width;
    if (payload != 0 && plane > (UINT64_MAX-header.keys_offset)/payload)
        throw std::runtime_error("archive key offset overflow");
    const std::uint64_t base=header.keys_offset+plane*payload;
    if(base>guard.n || payload>guard.n-base) throw std::runtime_error("truncated archive keys");
    using Candidate = std::pair<std::uint32_t, std::uint64_t>; std::priority_queue<Candidate> best;
    const auto* keys=static_cast<const std::uint8_t*>(mapping)+base;
    for (std::uint64_t i = 0; i < header.token_count; ++i) {
        Candidate c{popcount_xor(query, std::span<const std::uint8_t>(keys+i*header.packed_width,header.packed_width)), i};
        if (best.size() < top_k) best.push(c); else if (c < best.top()) { best.pop(); best.push(c); }
    }
    ScanResult result; result.backend = "cpu"; result.indices.resize(best.size()); result.distances.resize(best.size());
    for (std::size_t i = best.size(); i-- > 0;) { result.distances[i] = best.top().first; result.indices[i] = best.top().second; best.pop(); }
    result.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(); return result;
}

bool metal_available() { return false; }
std::vector<std::string> available_archive_backends() { return {"cpu"}; }
ScanResult scan_archive_metal(const std::filesystem::path& path, std::span<const std::uint8_t> query,
                              std::uint32_t layer, std::uint32_t head, std::size_t top_k) {
    (void)path;(void)query;(void)layer;(void)head;(void)top_k;
    throw std::runtime_error("Metal archive backend is unavailable on this platform");
}

} // namespace shadowrt
