#define TERNARY_GEMV_NO_MAIN
#include "../a53/ternary_gemv.c"

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
#include <sys/auxv.h>
#include <asm/hwcap.h>
#endif

enum { DOT_OUTPUT_TILE = 4, DOT_INPUT_TILE = 4 };

/* Layout: [4 output rows][4 input bytes], a row-major 4x4 tile per input block. */
static size_t dot_offset(size_t output, size_t input, size_t input_size) {
    size_t blocks = input_size / DOT_INPUT_TILE;
    return (output / DOT_OUTPUT_TILE) * blocks * 16
           + (input / DOT_INPUT_TILE) * 16
           + (output % DOT_OUTPUT_TILE) * 4 + input % DOT_INPUT_TILE;
}

static int8_t *repack_dotprod(const int8_t *source, size_t input_size, size_t output_size) {
    if (input_size % DOT_INPUT_TILE || output_size % DOT_OUTPUT_TILE) return NULL;
    int8_t *destination = malloc(input_size * output_size);
    if (!destination) return NULL;
    for (size_t output = 0; output < output_size; ++output)
        for (size_t input = 0; input < input_size; ++input)
            destination[dot_offset(output, input, input_size)] =
                source[weight_offset(output, input, output % OUTPUT_TILE, input_size)];
    return destination;
}

static uint8_t *repack_nibbles(const int8_t *source,size_t input_size,size_t output_size,int int4) {
    if (input_size % DOT_INPUT_TILE || output_size % DOT_OUTPUT_TILE) return NULL;
    size_t blocks = input_size / DOT_INPUT_TILE;
    uint8_t *destination = malloc(input_size * output_size / 2);
    if (!destination) return NULL;
    for (size_t output = 0; output < output_size; ++output)
        for (size_t input = 0; input < input_size; input += 2) {
            int8_t low = source[weight_offset(output, input, output % OUTPUT_TILE, input_size)];
            int8_t high = source[weight_offset(output, input + 1,
                                               output % OUTPUT_TILE, input_size)];
            size_t offset = (output / DOT_OUTPUT_TILE) * blocks * 8
                            + (input / DOT_INPUT_TILE) * 8
                            + (output % DOT_OUTPUT_TILE) * 2 + (input % DOT_INPUT_TILE) / 2;
            uint8_t low_code=int4 ? (uint8_t)low&15 : (uint8_t)(low+1);
            uint8_t high_code=int4 ? (uint8_t)high&15 : (uint8_t)(high+1);
            destination[offset]=low_code|(uint8_t)(high_code<<4);
        }
    return destination;
}

static void dotprod_scalar(const int8_t *weights, const int8_t *activation, int32_t *output,
                           size_t input_size, size_t output_size) {
    for (size_t row = 0; row < output_size; ++row) {
        int32_t sum = 0;
        for (size_t input = 0; input < input_size; ++input)
            sum += activation[input] * weights[dot_offset(row, input, input_size)];
        output[row] = sum;
    }
}

static int8_t nibble_weight(const uint8_t *weights,size_t output,size_t input,
                            size_t input_size,int int4) {
    size_t blocks = input_size / DOT_INPUT_TILE;
    size_t offset = (output / DOT_OUTPUT_TILE) * blocks * 8
                    + (input / DOT_INPUT_TILE) * 8
                    + (output % DOT_OUTPUT_TILE) * 2 + (input % DOT_INPUT_TILE) / 2;
    uint8_t packed = weights[offset];
    uint8_t code = input & 1 ? packed >> 4 : packed & 15;
    return int4 ? (int8_t)(code>=8 ? code-16 : code) : (int8_t)code-1;
}

static void nibble_scalar(const uint8_t *weights,const int8_t *activation,int32_t *output,
                          size_t input_size,size_t output_size,int int4) {
    for (size_t row = 0; row < output_size; ++row) {
        int32_t sum = 0;
        for (size_t input = 0; input < input_size; ++input)
            sum += activation[input]*nibble_weight(weights,row,input,input_size,int4);
        output[row] = sum;
    }
}

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
static int dotprod_available(void) {
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
}

static void dotprod_neon(const int8_t *weights, const int8_t *activation, int32_t *output,
                         size_t input_size, size_t output_size) {
    size_t blocks = input_size / DOT_INPUT_TILE;
    for (size_t output_base = 0; output_base < output_size; output_base += DOT_OUTPUT_TILE) {
        int32x4_t sum0 = vdupq_n_s32(0), sum1 = vdupq_n_s32(0);
        int32x4_t sum2 = vdupq_n_s32(0), sum3 = vdupq_n_s32(0);
        const int8_t *tile = weights + (output_base / DOT_OUTPUT_TILE) * blocks * 16;
        size_t block = 0;
#define DOT_BLOCK(SUM, INDEX) do {                                               \
            int32_t packed;                                                       \
            memcpy(&packed, activation + (INDEX) * DOT_INPUT_TILE, sizeof(packed)); \
            int8x16_t value = vreinterpretq_s8_s32(vdupq_n_s32(packed));          \
            SUM = vdotq_s32(SUM, vld1q_s8(tile + (INDEX) * 16), value);           \
        } while (0)
        for (; block + 4 <= blocks; block += 4) {
            DOT_BLOCK(sum0, block); DOT_BLOCK(sum1, block + 1);
            DOT_BLOCK(sum2, block + 2); DOT_BLOCK(sum3, block + 3);
        }
        for (; block < blocks; ++block) DOT_BLOCK(sum0, block);
#undef DOT_BLOCK
        vst1q_s32(output + output_base,
                  vaddq_s32(vaddq_s32(sum0, sum1), vaddq_s32(sum2, sum3)));
    }
}

static int8x16_t unpack_nibbles(uint8x8_t packed) {
    uint8x8_t low = vand_u8(packed, vdup_n_u8(15));
    uint8x8_t high = vshr_n_u8(packed, 4);
    uint8x8x2_t interleaved = vzip_u8(low, high);
    uint8x16_t code = vcombine_u8(interleaved.val[0], interleaved.val[1]);
    return vsubq_s8(vreinterpretq_s8_u8(code), vdupq_n_s8(1));
}
static int8x16_t unpack_int4(uint8x8_t packed) {
    uint8x8_t low=vand_u8(packed,vdup_n_u8(15)),high=vshr_n_u8(packed,4);
    uint8x8x2_t interleaved=vzip_u8(low,high);
    int8x16_t code=vreinterpretq_s8_u8(vcombine_u8(interleaved.val[0],interleaved.val[1]));
    return vshrq_n_s8(vshlq_n_s8(code,4),4);
}

static void nibble_neon(const uint8_t *weights,const int8_t *activation,int32_t *output,
                        size_t input_size,size_t output_size,int int4) {
    size_t blocks = input_size / DOT_INPUT_TILE;
    for (size_t output_base = 0; output_base < output_size; output_base += DOT_OUTPUT_TILE) {
        int32x4_t sum0 = vdupq_n_s32(0), sum1 = vdupq_n_s32(0);
        int32x4_t sum2 = vdupq_n_s32(0), sum3 = vdupq_n_s32(0);
        const uint8_t *tile = weights + (output_base / DOT_OUTPUT_TILE) * blocks * 8;
        size_t block = 0;
#define NIBBLE_BLOCK(SUM, INDEX) do {                                            \
            int32_t packed;                                                       \
            memcpy(&packed, activation + (INDEX) * DOT_INPUT_TILE, sizeof(packed)); \
            int8x16_t value = vreinterpretq_s8_s32(vdupq_n_s32(packed));          \
            uint8x8_t packed_weight=vld1_u8(tile+(INDEX)*8);                       \
            int8x16_t unpacked=int4 ? unpack_int4(packed_weight)                  \
                                     : unpack_nibbles(packed_weight);              \
            SUM=vdotq_s32(SUM,unpacked,value);                                    \
        } while (0)
        for (; block + 4 <= blocks; block += 4) {
            NIBBLE_BLOCK(sum0, block); NIBBLE_BLOCK(sum1, block + 1);
            NIBBLE_BLOCK(sum2, block + 2); NIBBLE_BLOCK(sum3, block + 3);
        }
        for (; block < blocks; ++block) NIBBLE_BLOCK(sum0, block);
#undef NIBBLE_BLOCK
        vst1q_s32(output + output_base,
                  vaddq_s32(vaddq_s32(sum0, sum1), vaddq_s32(sum2, sum3)));
    }
}
#else
static int dotprod_available(void) { return 0; }
static void dotprod_neon(const int8_t *weights, const int8_t *activation, int32_t *output,
                         size_t input_size, size_t output_size) {
    dotprod_scalar(weights, activation, output, input_size, output_size);
}
static void nibble_neon(const uint8_t *weights,const int8_t *activation,int32_t *output,
                        size_t input_size,size_t output_size,int int4) {
    nibble_scalar(weights,activation,output,input_size,output_size,int4);
}
#endif

int main(int argc, char **argv) {
    int nibble = argc > 1 && !strcmp(argv[1], "--nibble");
    int base = nibble ? 2 : 1;
    int from_shdw = argc > base && !strcmp(argv[base], "--shdw");
    size_t input_size = from_shdw ? 0 : (argc > base ? strtoull(argv[base], NULL, 10) : 1536);
    size_t output_size = from_shdw ? 0 : (argc > base + 1
        ? strtoull(argv[base + 1], NULL, 10) : 4224);
    size_t iterations = from_shdw ? (argc > base + 3
        ? strtoull(argv[base + 3], NULL, 10) : 100) : (argc > base + 2
        ? strtoull(argv[base + 2], NULL, 10) : 100);
    if (from_shdw && argc < base + 3) {
        fprintf(stderr, "usage: %s [--nibble] --shdw MODEL TENSOR [ITERATIONS]\n", argv[0]);
        return 2;
    }
    uint32_t weight_kind=4;
    int8_t *input_major=from_shdw
        ? load_shdw_weights(argv[base+1],argv[base+2],&input_size,&output_size,&weight_kind) : NULL;
    int int4=weight_kind==6;
    if (!input_size || !output_size || !iterations || input_size % DOT_INPUT_TILE
        || output_size % OUTPUT_TILE) {
        fprintf(stderr, "dimensions must be positive multiples of input=4/output=16\n");
        free(input_major); return 2;
    }
    size_t weight_count = input_size * output_size;
    if (!input_major) {
        input_major = malloc(weight_count);
        if (!input_major) return 3;
        for (size_t index = 0; index < weight_count; ++index)
            input_major[index] = (int8_t)(next_random() % 3) - 1;
    }
    int8_t *weights = nibble ? NULL : repack_dotprod(input_major, input_size, output_size);
    uint8_t *nibble_weights=nibble ? repack_nibbles(input_major,input_size,output_size,int4) : NULL;
    free(input_major);
    int8_t *activation = malloc(input_size);
    int32_t *reference = malloc(output_size * sizeof(*reference));
    int32_t *actual = malloc(output_size * sizeof(*actual));
    if ((!weights && !nibble_weights) || !activation || !reference || !actual) return 3;
    for (size_t input = 0; input < input_size; ++input)
        activation[input] = (int8_t)(next_random() % 255) - 127;
    if (nibble) {
        nibble_scalar(nibble_weights,activation,reference,input_size,output_size,int4);
        nibble_neon(nibble_weights,activation,actual,input_size,output_size,int4);
    } else {
        dotprod_scalar(weights, activation, reference, input_size, output_size);
        dotprod_neon(weights, activation, actual, input_size, output_size);
    }
    if (memcmp(reference, actual, output_size * sizeof(*actual))) {
        fprintf(stderr, "DotProd result differs from scalar reference\n"); return 4;
    }
    for (size_t warmup = 0; warmup < 5; ++warmup)
        if (nibble) nibble_neon(nibble_weights,activation,actual,input_size,output_size,int4);
        else dotprod_neon(weights, activation, actual, input_size, output_size);
    double start = seconds();
    for (size_t iteration = 0; iteration < iterations; ++iteration)
        if (nibble) nibble_neon(nibble_weights,activation,actual,input_size,output_size,int4);
        else dotprod_neon(weights, activation, actual, input_size, output_size);
    double elapsed = seconds() - start;
    int64_t checksum = 0;
    for (size_t output = 0; output < output_size; ++output) checksum += actual[output];
    printf("kernel=%s hw_dotprod=%d source=%s input=%zu output=%zu weights=%.3f_MiB "
           "time=%.3f_us checksum=%" PRId64 "\n",
#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
           nibble ? (int4 ? "a55_sdot_int4" : "a55_sdot_ternary_nibble")
                  : (int4 ? "a55_sdot_int4_bytes" : "a55_sdot_int8_trit"),
#else
           nibble ? "scalar_nibble_layout" : "scalar_dotprod_layout",
#endif
           dotprod_available(), from_shdw ? argv[base + 2] : "random", input_size, output_size,
           weight_count / (nibble ? 2.0 * 1048576.0 : 1048576.0),
           elapsed * 1e6 / iterations, checksum);
    free(actual); free(reference); free(activation); free(nibble_weights); free(weights);
    return 0;
}
