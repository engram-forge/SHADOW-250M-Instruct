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
