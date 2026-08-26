#define _POSIX_C_SOURCE 200809L
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

enum { OUTPUT_TILE = 16, PARTIAL_INPUTS = 128 };

/* Layout: [output_tile][input][16 output lanes]. */
static size_t weight_offset(size_t output, size_t input_index, size_t lane,
                            size_t input_size) {
    return (output / OUTPUT_TILE) * input_size * OUTPUT_TILE
           + input_index * OUTPUT_TILE + lane;
}

void ternary_gemv_scalar(const int8_t *weights, const int8_t *activation,
                         int32_t *output, size_t input_size, size_t output_size) {
    memset(output, 0, output_size * sizeof(*output));
    for (size_t output_base = 0; output_base < output_size; output_base += OUTPUT_TILE) {
        size_t lanes = output_size - output_base;
        if (lanes > OUTPUT_TILE) lanes = OUTPUT_TILE;
        for (size_t input = 0; input < input_size; ++input) {
            const int8_t *weight = weights + weight_offset(output_base, input, 0, input_size);
            int32_t value = activation[input];
            for (size_t lane = 0; lane < lanes; ++lane)
                output[output_base + lane] += value * weight[lane];
        }
    }
}

#if defined(__aarch64__)
void ternary_gemv_neon(const int8_t *weights, const int8_t *activation,
                       int32_t *output, size_t input_size, size_t output_size) {
    size_t vector_outputs = output_size & ~(size_t)(OUTPUT_TILE - 1);
    for (size_t output_base = 0; output_base < vector_outputs; output_base += OUTPUT_TILE) {
        int32x4_t total0 = vdupq_n_s32(0), total1 = vdupq_n_s32(0);
        int32x4_t total2 = vdupq_n_s32(0), total3 = vdupq_n_s32(0);
        const int8_t *tile = weights + (output_base / OUTPUT_TILE) * input_size * OUTPUT_TILE;
        for (size_t chunk = 0; chunk < input_size; chunk += PARTIAL_INPUTS) {
            size_t end = chunk + PARTIAL_INPUTS;
            if (end > input_size) end = input_size;
            int16x8_t low0 = vdupq_n_s16(0), high0 = vdupq_n_s16(0);
            int16x8_t low1 = vdupq_n_s16(0), high1 = vdupq_n_s16(0);
            int16x8_t low2 = vdupq_n_s16(0), high2 = vdupq_n_s16(0);
            int16x8_t low3 = vdupq_n_s16(0), high3 = vdupq_n_s16(0);
            size_t input = chunk;
#define ACCUMULATE(SUFFIX, INDEX) do {                                           \
                int8x16_t weight = vld1q_s8(tile + (INDEX) * OUTPUT_TILE);       \
                int8x16_t value = vdupq_n_s8(activation[(INDEX)]);               \
                low##SUFFIX = vmlal_s8(low##SUFFIX, vget_low_s8(weight),         \
                                       vget_low_s8(value));                       \
                high##SUFFIX = vmlal_high_s8(high##SUFFIX, weight, value);       \
            } while (0)
            for (; input + 4 <= end; input += 4) {
                ACCUMULATE(0, input);
                ACCUMULATE(1, input + 1);
                ACCUMULATE(2, input + 2);
                ACCUMULATE(3, input + 3);
            }
            for (; input < end; ++input) ACCUMULATE(0, input);
#undef ACCUMULATE
            int16x8_t low = vaddq_s16(vaddq_s16(low0, low1), vaddq_s16(low2, low3));
            int16x8_t high = vaddq_s16(vaddq_s16(high0, high1), vaddq_s16(high2, high3));
            total0 = vaddw_s16(total0, vget_low_s16(low));
            total1 = vaddw_high_s16(total1, low);
            total2 = vaddw_s16(total2, vget_low_s16(high));
            total3 = vaddw_high_s16(total3, high);
        }
        vst1q_s32(output + output_base, total0);
        vst1q_s32(output + output_base + 4, total1);
        vst1q_s32(output + output_base + 8, total2);
        vst1q_s32(output + output_base + 12, total3);
    }
    if (vector_outputs != output_size) {
        int32_t tail[OUTPUT_TILE] = {0};
        ternary_gemv_scalar(weights + (vector_outputs / OUTPUT_TILE) * input_size * OUTPUT_TILE,
                            activation, tail, input_size, output_size - vector_outputs);
        memcpy(output + vector_outputs, tail,
               (output_size - vector_outputs) * sizeof(*output));
    }
}
#else
void ternary_gemv_neon(const int8_t *weights, const int8_t *activation,
                       int32_t *output, size_t input_size, size_t output_size) {
    ternary_gemv_scalar(weights, activation, output, input_size, output_size);
}
#endif

static uint32_t random_state = 1;
static uint32_t next_random(void) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return random_state;
}

static double seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + value.tv_nsec * 1e-9;
}

static int read_exact(FILE *stream, void *destination, size_t size) {
    return fread(destination, 1, size, stream) == size;
}

static int skip_bytes(FILE *stream, uint64_t size) {
    while (size) {
        long chunk = size > 0x3fffffffU ? 0x3fffffffL : (long)size;
        if (fseek(stream, chunk, SEEK_CUR)) return 0;
        size -= (uint64_t)chunk;
    }
    return 1;
}

static int read_dimensions(FILE *stream, uint32_t *dimensions, uint64_t *count) {
    uint32_t rank;
    if (!read_exact(stream, &rank, sizeof(rank)) || rank > 16) return 0;
    *count = 1;
    for (uint32_t axis = 0; axis < rank; ++axis) {
        uint32_t value;
        if (!read_exact(stream, &value, sizeof(value))) return 0;
        if (value && *count > UINT64_MAX / value) return 0;
        *count *= value;
    }
    if (dimensions) *dimensions = rank;
    return 1;
}

static int skip_record(FILE *stream, uint32_t kind) {
    if (kind == 0 || kind == 5) {
        uint64_t count;
        if (!read_dimensions(stream, NULL, &count)) return 0;
        return skip_bytes(stream, count * (kind == 0 ? 4U : 2U));
    }
    if (kind == 1) {
        uint32_t shape[4];
        if (!read_exact(stream, shape, sizeof(shape)) || !shape[2]) return 0;
        uint64_t output = shape[0], input = shape[1], group = shape[2], stages = shape[3];
        uint64_t groups = input / group, padded_output = (output + 63U) & ~63U;
        uint64_t chunks = padded_output / 64U;
        return skip_bytes(stream, stages * group * 16U * 4U
                         + stages * chunks * groups * 32U + padded_output * 4U);
    }
    if (kind == 3 || kind == 4 || kind == 6) {
        uint32_t shape[2];
        if (!read_exact(stream, shape, sizeof(shape))) return 0;
        uint64_t symbols = kind == 3 ? (uint64_t)shape[0] * shape[1] / 4U
            : kind == 4 ? (uint64_t)shape[0] * ((shape[1] + 4U) / 5U)
                        : (uint64_t)shape[0] * ((shape[1] + 1U) / 2U);
        return skip_bytes(stream, symbols + (uint64_t)shape[0] * 4U);
    }
    return 0;
}

static int8_t *load_shdw_weights(const char *path,const char *wanted,
                                 size_t *input_size,size_t *output_size,uint32_t *weight_kind) {
    FILE *stream = fopen(path, "rb");
    char magic[4];
    uint32_t version, records;
    if (!stream || !read_exact(stream, magic, sizeof(magic))
        || memcmp(magic, "SHDW", 4)
        || !read_exact(stream, &version, sizeof(version))
        || !read_exact(stream, &records, sizeof(records)) || version != 1) {
        fprintf(stderr, "invalid SHDW file %s\n", path);
        if (stream) fclose(stream);
        return NULL;
    }
    for (uint32_t record = 0; record < records; ++record) {
        uint32_t name_size, kind;
        if (!read_exact(stream, &name_size, sizeof(name_size)) || name_size > (1U << 20)) break;
        char *name = malloc((size_t)name_size + 1);
        if (!name || !read_exact(stream, name, name_size)
            || !read_exact(stream, &kind, sizeof(kind))) {
            free(name);
            break;
        }
        name[name_size] = 0;
        int selected = !strcmp(name, wanted);
        free(name);
        if (!selected) {
            if (!skip_record(stream, kind)) break;
            continue;
        }
        if (kind != 4 && kind != 6) {
            fprintf(stderr, "tensor %s has kind %u, expected ternary 4 or INT4 6\n",
                    wanted, kind);
            fclose(stream);
            return NULL;
        }
        uint32_t shape[2];
        if (!read_exact(stream, shape, sizeof(shape)) || shape[0] % OUTPUT_TILE) break;
        *output_size = shape[0]; *input_size = shape[1];
        *weight_kind=kind;
        size_t packed_row = kind==4 ? (*input_size+4)/5 : (*input_size+1)/2;
        if (*input_size && *output_size > SIZE_MAX / *input_size) break;
        int8_t *weights = calloc(*input_size * *output_size, 1);
        uint8_t *packed = malloc(packed_row);
        if (!weights || !packed) { free(weights); free(packed); break; }
        for (size_t output = 0; output < *output_size; ++output) {
            if (!read_exact(stream, packed, packed_row)) {
                free(weights); free(packed); fclose(stream); return NULL;
            }
            for (size_t input = 0; input < *input_size; ++input) {
                uint8_t value;
                if (kind==4) {
                    value=packed[input/5];
                    for (size_t trit=0;trit<input%5;++trit) value/=3;
                    value=(uint8_t)((int)(value%3)-1);
                } else {
                    value=input&1 ? packed[input/2]>>4 : packed[input/2]&15;
                    if (value>=8) value=(uint8_t)(value-16);
                }
                weights[(output / OUTPUT_TILE) * *input_size * OUTPUT_TILE
                        + input * OUTPUT_TILE + output % OUTPUT_TILE] = (int8_t)value;
            }
        }
        free(packed);
        if (!skip_bytes(stream, (uint64_t)*output_size * 4U)) { free(weights); break; }
        fclose(stream);
        return weights;
    }
    fprintf(stderr, "quantized tensor %s not found in %s\n", wanted, path);
    fclose(stream);
    return NULL;
}

#ifndef TERNARY_GEMV_NO_MAIN
int main(int argc, char **argv) {
    int from_shdw = argc > 1 && !strcmp(argv[1], "--shdw");
    size_t input_size = from_shdw ? 0 : (argc > 1 ? strtoull(argv[1], NULL, 10) : 1536);
    size_t output_size = from_shdw ? 0 : (argc > 2 ? strtoull(argv[2], NULL, 10) : 4224);
    size_t iterations = from_shdw ? (argc > 4 ? strtoull(argv[4], NULL, 10) : 100)
                                  : (argc > 3 ? strtoull(argv[3], NULL, 10) : 100);
    if (from_shdw && argc < 4) {
        fprintf(stderr, "usage: %s --shdw MODEL SHDW_TENSOR [ITERATIONS]\n", argv[0]);
        return 2;
    }
    uint32_t weight_kind=4;
    int8_t *weights = from_shdw
        ? load_shdw_weights(argv[2],argv[3],&input_size,&output_size,&weight_kind) : NULL;
    if (from_shdw && !weights) return 2;
    if (!input_size || !output_size || !iterations || output_size % OUTPUT_TILE) {
        fprintf(stderr, "input/output/iterations must be positive; output must be a multiple of 16\n");
        free(weights);
        return 2;
    }
    size_t weight_count = input_size * output_size;
    if (!weights) weights = malloc(weight_count);
    int8_t *activation = malloc(input_size);
    int32_t *reference = malloc(output_size * sizeof(*reference));
    int32_t *actual = malloc(output_size * sizeof(*actual));
    if (!weights || !activation || !reference || !actual) return 3;
    if (!from_shdw)
        for (size_t index = 0; index < weight_count; ++index)
            weights[index] = (int8_t)(next_random() % 3) - 1;
    for (size_t index = 0; index < input_size; ++index)
        activation[index] = (int8_t)(next_random() % 255) - 127;

    ternary_gemv_scalar(weights, activation, reference, input_size, output_size);
    ternary_gemv_neon(weights, activation, actual, input_size, output_size);
    if (memcmp(reference, actual, output_size * sizeof(*actual))) {
        for (size_t index = 0; index < output_size; ++index)
            if (reference[index] != actual[index]) {
                fprintf(stderr, "mismatch at %zu: %" PRId32 " != %" PRId32 "\n",
                        index, reference[index], actual[index]);
                break;
            }
        return 4;
    }
    for (size_t warmup = 0; warmup < 5; ++warmup)
        ternary_gemv_neon(weights, activation, actual, input_size, output_size);
    double start = seconds();
    for (size_t iteration = 0; iteration < iterations; ++iteration)
        ternary_gemv_neon(weights, activation, actual, input_size, output_size);
    double elapsed = seconds() - start;
    int64_t checksum = 0;
    for (size_t index = 0; index < output_size; ++index) checksum += actual[index];
    printf("kernel=%s source=%s input=%zu output=%zu weights=%.3f_MiB time=%.3f_us checksum=%" PRId64 "\n",
#if defined(__aarch64__)
           "a53_neon_int8_trit",
#else
           "scalar_int8_trit",
#endif
           from_shdw ? argv[3] : "random", input_size, output_size, weight_count / 1048576.0,
           elapsed * 1e6 / iterations, checksum);
    free(actual); free(reference); free(activation); free(weights);
    return 0;
}
#endif
