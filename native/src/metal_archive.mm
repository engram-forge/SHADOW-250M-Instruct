#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "shadow/archive.hpp"
#include "shadow/model.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <mach-o/dyld.h>
#include <queue>
#include <mutex>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace shadowrt {
namespace {

constexpr std::uint32_t chunk_size = 4096, group_width = 256, max_k = 32;

const char* source = R"METAL(
#include <metal_stdlib>
using namespace metal;

constant uint CHUNK = 4096;
constant uint WIDTH = 256;
constant uint MAX_K = 32;

kernel void hamming_topk(
    device const uchar* file [[buffer(0)]],
    device const uchar* query [[buffer(1)]],
    device uint* output [[buffer(2)]],
    constant ulong& key_offset [[buffer(3)]],
    constant ulong& token_count [[buffer(4)]],
    constant uint& packed_width [[buffer(5)]],
    constant uint& take [[buffer(6)]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    threadgroup uint candidates[CHUNK];
    const ulong begin = ulong(group) * CHUNK;
    for (uint local = lane; local < CHUNK; local += WIDTH) {
        const ulong index = begin + local;
        uint distance = 0;
        if (index < token_count) {
            const ulong base = key_offset + index * packed_width;
            for (uint byte = 0; byte < packed_width; ++byte)
                distance += popcount(uint(file[base + byte] ^ query[byte]));
            candidates[local] = (distance << 22) | local;
        } else candidates[local] = 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 2; k <= CHUNK; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            for (uint i = lane; i < CHUNK; i += WIDTH) {
                uint ixj = i ^ j;
                if (ixj > i) {
                    bool ascending = (i & k) == 0;
                    uint a = candidates[i], b = candidates[ixj];
                    if ((a > b) == ascending) { candidates[i] = b; candidates[ixj] = a; }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (lane < take) output[group * MAX_K + lane] = candidates[lane];
}
)METAL";

ArchiveHeader load_header(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary); ArchiveHeader h{}; in.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!in || std::memcmp(h.magic, "SHARKV1", 7) || h.version != 1 || h.header_bytes != sizeof(h))
        throw std::runtime_error("invalid SHADOW KV archive");
    return h;
}

void validate(const ArchiveHeader& h, std::span<const std::uint8_t> query, std::uint32_t layer, std::uint32_t head, std::size_t k) {
    if (h.token_count == 0) throw std::runtime_error("cannot scan an empty archive");
    if (layer >= h.layers || head >= h.heads || query.size() != h.packed_width || k == 0 || k > max_k)
        throw std::runtime_error("archive scan shape mismatch (Metal top-k supports 1..32)");
}

id<MTLLibrary> load_library(id<MTLDevice> device, NSError** error) {
    std::uint32_t size = 0; _NSGetExecutablePath(nullptr, &size);
    std::vector<char> executable(size);
    if (_NSGetExecutablePath(executable.data(), &size) == 0) {
        auto path = std::filesystem::weakly_canonical(executable.data());
        path += ".metallib";
        if (std::filesystem::exists(path)) {
            NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
            if (id<MTLLibrary> library = [device newLibraryWithURL:url error:error]) return library;
        }
    }
    NSString* text = [NSString stringWithUTF8String:source];
    return [device newLibraryWithSource:text options:nil error:error];
}

struct MetalContext {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeline = nil;
};

MetalContext& metal_context() {
    static MetalContext context = [] {
        MetalContext value; value.device = MTLCreateSystemDefaultDevice();
        if (!value.device) return value;
        NSError* error = nil; id<MTLLibrary> library = load_library(value.device, &error);
        if (!library) throw std::runtime_error([[error localizedDescription] UTF8String]);
        value.pipeline = [value.device newComputePipelineStateWithFunction:[library newFunctionWithName:@"hamming_topk"] error:&error];
        if (!value.pipeline) throw std::runtime_error([[error localizedDescription] UTF8String]);
        value.queue = [value.device newCommandQueue]; return value;
    }();
    return context;
}

struct ArchiveMetalCache {
    std::filesystem::path path;
    void* mapping = MAP_FAILED;
    std::size_t length = 0;
    id<MTLBuffer> file = nil;
    id<MTLBuffer> query = nil;
    id<MTLBuffer> output = nil;
    std::size_t output_bytes = 0;
    std::mutex mutex;

    ~ArchiveMetalCache() {
        output = nil; query = nil; file = nil;
        if (mapping != MAP_FAILED) munmap(mapping, length);
    }

    void bind(const std::filesystem::path& requested, id<MTLDevice> device,
              std::size_t query_bytes, std::size_t required_output_bytes) {
        if (path != requested || !file) {
            output = nil; query = nil; file = nil; output_bytes = 0;
            if (mapping != MAP_FAILED) { munmap(mapping, length); mapping = MAP_FAILED; length = 0; }
            const int fd = open(requested.c_str(), O_RDONLY);
            if (fd < 0) throw std::runtime_error("cannot open archive");
            struct stat st{};
            if (fstat(fd, &st)) { close(fd); throw std::runtime_error("cannot stat archive"); }
            length = static_cast<std::size_t>(st.st_size);
            mapping = mmap(nullptr, length, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
            if (mapping == MAP_FAILED) throw std::runtime_error("cannot mmap archive");
            file = [device newBufferWithBytesNoCopy:mapping length:length
                                             options:MTLResourceStorageModeShared deallocator:nil];
            if (!file) { munmap(mapping, length); mapping = MAP_FAILED; length = 0; throw std::runtime_error("Metal cannot bind archive mapping"); }
            path = requested;
        }
        if (!query || query.length < query_bytes)
            query = [device newBufferWithLength:query_bytes options:MTLResourceStorageModeShared];
        if (!output || output_bytes < required_output_bytes) {
            output = [device newBufferWithLength:required_output_bytes options:MTLResourceStorageModeShared];
            output_bytes = required_output_bytes;
        }
    }
};

ArchiveMetalCache& archive_cache() { static ArchiveMetalCache cache; return cache; }

} // namespace

ScanResult scan_archive_cpu(const std::filesystem::path& path, std::span<const std::uint8_t> query,
                            std::uint32_t layer, std::uint32_t head, std::size_t top_k) {
    const auto started = std::chrono::steady_clock::now(); const auto h = load_header(path); validate(h, query, layer, head, top_k);
    std::ifstream in(path, std::ios::binary);
    const std::uint64_t base = h.keys_offset + ((static_cast<std::uint64_t>(layer) * h.heads + head) * h.token_count * h.packed_width);
    in.seekg(static_cast<std::streamoff>(base)); std::vector<std::uint8_t> key(h.packed_width);
    using Candidate = std::pair<std::uint32_t, std::uint64_t>; std::priority_queue<Candidate> best;
    for (std::uint64_t i = 0; i < h.token_count; ++i) {
        in.read(reinterpret_cast<char*>(key.data()), key.size()); if (!in) throw std::runtime_error("truncated archive keys");
        Candidate c{popcount_xor(query, key), i}; if (best.size() < top_k) best.push(c); else if (c < best.top()) { best.pop(); best.push(c); }
    }
    ScanResult r; r.backend = "cpu"; r.indices.resize(best.size()); r.distances.resize(best.size());
    for (std::size_t i = best.size(); i-- > 0;) { r.distances[i] = best.top().first; r.indices[i] = best.top().second; best.pop(); }
    r.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(); return r;
}

bool metal_available() {
    @autoreleasepool {
        try { return metal_context().device != nil; }
        catch (...) { return false; }
    }
}
std::vector<std::string> available_archive_backends() {
    return metal_available() ? std::vector<std::string>{"cpu", "metal"}
                             : std::vector<std::string>{"cpu"};
}

ScanResult scan_archive_metal(const std::filesystem::path& path, std::span<const std::uint8_t> query,
                              std::uint32_t layer, std::uint32_t head, std::size_t top_k) {
    @autoreleasepool {
        const auto started = std::chrono::steady_clock::now(); const auto h = load_header(path); validate(h, query, layer, head, top_k);
        auto& context = metal_context(); id<MTLDevice> device = context.device; if (!device) return scan_archive_cpu(path, query, layer, head, top_k);
        id<MTLComputePipelineState> pipeline = context.pipeline;
        const std::uint64_t groups = (h.token_count + chunk_size - 1) / chunk_size;
        ArchiveMetalCache transient;
        const bool disable_cache = std::getenv("SHADOW_METAL_CACHE") && std::string_view(std::getenv("SHADOW_METAL_CACHE")) == "0";
        auto& cache = disable_cache ? transient : archive_cache(); std::lock_guard cache_lock(cache.mutex);
        cache.bind(path, device, query.size(), groups * max_k * sizeof(std::uint32_t));
        std::memcpy(cache.query.contents, query.data(), query.size());
        id<MTLBuffer> file = cache.file, qbuf = cache.query, out = cache.output;
        const std::uint64_t key_offset = h.keys_offset + ((static_cast<std::uint64_t>(layer) * h.heads + head) * h.token_count * h.packed_width);
        const std::uint32_t take = static_cast<std::uint32_t>(top_k);
        id<MTLCommandBuffer> command = [context.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder]; [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:file offset:0 atIndex:0]; [encoder setBuffer:qbuf offset:0 atIndex:1]; [encoder setBuffer:out offset:0 atIndex:2];
        [encoder setBytes:&key_offset length:sizeof(key_offset) atIndex:3]; [encoder setBytes:&h.token_count length:sizeof(h.token_count) atIndex:4];
        [encoder setBytes:&h.packed_width length:sizeof(h.packed_width) atIndex:5]; [encoder setBytes:&take length:sizeof(take) atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(group_width, 1, 1)]; [encoder endEncoding];
        [command commit]; [command waitUntilCompleted];
        if (command.status == MTLCommandBufferStatusError) throw std::runtime_error([[command.error localizedDescription] UTF8String]);
        using Candidate = std::pair<std::uint32_t, std::uint64_t>; std::priority_queue<Candidate> best;
        auto* values = static_cast<const std::uint32_t*>(out.contents);
        for (std::uint64_t group = 0; group < groups; ++group) for (std::size_t rank = 0; rank < top_k; ++rank) {
            const auto packed = values[group * max_k + rank]; if (packed == 0xffffffffu) continue;
            Candidate c{packed >> 22, group * chunk_size + (packed & ((1u << 22) - 1))};
            if (best.size() < top_k) best.push(c); else if (c < best.top()) { best.pop(); best.push(c); }
        }
        ScanResult r; r.backend = "metal"; r.indices.resize(best.size()); r.distances.resize(best.size());
        for (std::size_t i = best.size(); i-- > 0;) { r.distances[i] = best.top().first; r.indices[i] = best.top().second; best.pop(); }
        r.milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
        return r;
    }
}

} // namespace shadowrt
