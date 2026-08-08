#ifndef TINYPOET_ENGINE_H
#define TINYPOET_ENGINE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Engine Constants & Upper Bounds (Tightened to SRAM budget)
#define MAX_LAYERS 2
#define MAX_HEADS 4
#define MAX_EMBD 80
#define MAX_BLOCK_SIZE 64
#define MAX_VOCAB_SIZE 256

typedef enum {
    WEIGHT_FMT_FP32 = 0,
    WEIGHT_FMT_INT4 = 1,
    WEIGHT_FMT_TERNARY_2BIT = 2,
    WEIGHT_FMT_TERNARY_BASE3 = 3
} WeightFormat;

typedef struct {
    uint32_t n_layer;
    uint32_t n_head;
    uint32_t n_embd;
    uint32_t block_size;
    uint32_t vocab_size;
    WeightFormat format;
} ESP32LLMConfigC;

typedef struct {
    ESP32LLMConfigC config;
    
    // KV Cache: shape [n_layer, 2 (k,v), block_size, n_embd] = 81.92 KB
    float kv_cache[MAX_LAYERS][2][MAX_BLOCK_SIZE][MAX_EMBD];
    
    // Working activations buffer (Zero heap allocations during step)
    float x_buf[MAX_EMBD];
    float norm_buf[MAX_EMBD];
    float q_buf[MAX_EMBD];
    float k_buf[MAX_EMBD];
    float v_buf[MAX_EMBD];
    float attn_scores[MAX_BLOCK_SIZE];
    float mlp_hidden[4 * MAX_EMBD];
    float logits[MAX_VOCAB_SIZE];
    
    const uint8_t *model_weights_ptr; // Pointer to Flash memory-mapped weights
} ESP32LLMEngine;

// Engine API Functions
void esp32_llm_init(ESP32LLMEngine *engine, const ESP32LLMConfigC *config, const uint8_t *weights_ptr);
void esp32_llm_reset_kv_cache(ESP32LLMEngine *engine);
void esp32_llm_forward_step(ESP32LLMEngine *engine, int token_id, int pos, float *out_logits);

// Packing / Unpacking Helper Functions
void unpack_2bit_block(const uint8_t *packed, int8_t *out_weights, int num_weights);
void unpack_base3_block(const uint8_t *packed, int8_t *out_weights, int num_weights);
void unpack_int4_block(const uint8_t *packed, int8_t *out_weights, int num_weights);

// Math primitives (SIMD / DSP friendly)
void matvec_fp32(const float *mat, const float *vec, float *out, int rows, int cols);
void matvec_ternary_2bit(const uint8_t *packed_w, float scale, const float *vec, float *out, int rows, int cols);
void matvec_ternary_base3(const uint8_t *packed_w, float scale, const float *vec, float *out, int rows, int cols);

#ifdef __cplusplus
}
#endif

#endif // TINYPOET_ENGINE_H
