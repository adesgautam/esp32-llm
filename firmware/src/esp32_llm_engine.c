#include "esp32_llm_engine.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

static inline float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

static inline float square(float x) {
    return x * x;
}

// 2-Bit Ternary Unpacker: 4 weights per byte
// Mapping: 0 -> -1, 1 -> 0, 2 -> +1
void unpack_2bit_block(const uint8_t *packed, int8_t *out_weights, int num_weights) {
    static const int8_t inv_map[4] = {-1, 0, 1, 0};
    int idx = 0;
    int bytes = (num_weights + 3) / 4;
    for (int i = 0; i < bytes; i++) {
        uint8_t b = packed[i];
        if (idx < num_weights) out_weights[idx++] = inv_map[b & 0x03];
        if (idx < num_weights) out_weights[idx++] = inv_map[(b >> 2) & 0x03];
        if (idx < num_weights) out_weights[idx++] = inv_map[(b >> 4) & 0x03];
        if (idx < num_weights) out_weights[idx++] = inv_map[(b >> 6) & 0x03];
    }
}

// Base-3 Arithmetic Ternary Unpacker: 5 weights per byte (3^5 = 243 <= 256)
// Mapping: 0 -> -1, 1 -> 0, 2 -> +1
void unpack_base3_block(const uint8_t *packed, int8_t *out_weights, int num_weights) {
    static const int8_t inv_map[3] = {-1, 0, 1};
    int idx = 0;
    int bytes = (num_weights + 4) / 5;
    for (int i = 0; i < bytes; i++) {
        uint32_t val = packed[i];
        uint32_t v0 = val % 3; val /= 3;
        uint32_t v1 = val % 3; val /= 3;
        uint32_t v2 = val % 3; val /= 3;
        uint32_t v3 = val % 3; val /= 3;
        uint32_t v4 = val % 3;
        
        if (idx < num_weights) out_weights[idx++] = inv_map[v0];
        if (idx < num_weights) out_weights[idx++] = inv_map[v1];
        if (idx < num_weights) out_weights[idx++] = inv_map[v2];
        if (idx < num_weights) out_weights[idx++] = inv_map[v3];
        if (idx < num_weights) out_weights[idx++] = inv_map[v4];
    }
}

// INT4 Unpacker: 2 weights per byte [-8, 7]
void unpack_int4_block(const uint8_t *packed, int8_t *out_weights, int num_weights) {
    int idx = 0;
    int bytes = (num_weights + 1) / 2;
    for (int i = 0; i < bytes; i++) {
        uint8_t b = packed[i];
        int8_t w0 = (int8_t)(b & 0x0F) - 8;
        int8_t w1 = (int8_t)((b >> 4) & 0x0F) - 8;
        if (idx < num_weights) out_weights[idx++] = w0;
        if (idx < num_weights) out_weights[idx++] = w1;
    }
}

void matvec_fp32(const float *mat, const float *vec, float *out, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float sum = 0.0f;
        const float *row_ptr = mat + r * cols;
        for (int c = 0; c < cols; c++) {
            sum += row_ptr[c] * vec[c];
        }
        out[r] = sum;
    }
}

void matvec_ternary_2bit(const uint8_t *packed_w, float scale, const float *vec, float *out, int rows, int cols) {
    int bytes_per_row = (cols + 3) / 4;
    int8_t w_buf[MAX_EMBD * 4];
    
    for (int r = 0; r < rows; r++) {
        const uint8_t *row_packed = packed_w + r * bytes_per_row;
        unpack_2bit_block(row_packed, w_buf, cols);
        float acc = 0.0f;
        for (int c = 0; c < cols; c++) {
            if (w_buf[c] == 1) acc += vec[c];
            else if (w_buf[c] == -1) acc -= vec[c];
        }
        out[r] = acc * scale;
    }
}

void matvec_ternary_base3(const uint8_t *packed_w, float scale, const float *vec, float *out, int rows, int cols) {
    int bytes_per_row = (cols + 4) / 5;
    int8_t w_buf[MAX_EMBD * 4];
    
    for (int r = 0; r < rows; r++) {
        const uint8_t *row_packed = packed_w + r * bytes_per_row;
        unpack_base3_block(row_packed, w_buf, cols);
        float acc = 0.0f;
        for (int c = 0; c < cols; c++) {
            if (w_buf[c] == 1) acc += vec[c];
            else if (w_buf[c] == -1) acc -= vec[c];
        }
        out[r] = acc * scale;
    }
}

void layer_norm(const float *x, const float *gamma, const float *beta, float *out, int dim) {
    float mean = 0.0f;
    for (int i = 0; i < dim; i++) mean += x[i];
    mean /= dim;
    
    float var = 0.0f;
    for (int i = 0; i < dim; i++) var += square(x[i] - mean);
    var /= dim;
    
    float inv_std = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < dim; i++) {
        out[i] = (x[i] - mean) * inv_std * (gamma ? gamma[i] : 1.0f) + (beta ? beta[i] : 0.0f);
    }
}

void esp32_llm_init(ESP32LLMEngine *engine, const ESP32LLMConfigC *config, const uint8_t *weights_ptr) {
    memset(engine, 0, sizeof(ESP32LLMEngine));
    engine->config = *config;
    engine->model_weights_ptr = weights_ptr;
    esp32_llm_reset_kv_cache(engine);
}

void esp32_llm_reset_kv_cache(ESP32LLMEngine *engine) {
    memset(engine->kv_cache, 0, sizeof(engine->kv_cache));
}

void esp32_llm_forward_step(ESP32LLMEngine *engine, int token_id, int pos, float *out_logits) {
    uint32_t n_layer = engine->config.n_layer;
    uint32_t n_head = engine->config.n_head;
    uint32_t n_embd = engine->config.n_embd;
    uint32_t block_size = engine->config.block_size;
    uint32_t vocab_size = engine->config.vocab_size;
    
    // Header is 7 * uint32 = 28 bytes
    const float* w = (const float*)(engine->model_weights_ptr + 28);
    
    // Embeddings
    const float* wte = w; w += vocab_size * n_embd;
    const float* wpe = w; w += block_size * n_embd;
    
    // 1. Token + Positional Embedding
    for (uint32_t i = 0; i < n_embd; i++) {
        engine->x_buf[i] = wte[token_id * n_embd + i] + wpe[pos * n_embd + i];
    }
    
    for (uint32_t l = 0; l < n_layer; l++) {
        const float* ln_1_w = w; w += n_embd;
        const float* ln_1_b = w; w += n_embd;
        // Skip causal mask buffer: attn.bias [1, 1, block_size, block_size]
        w += block_size * block_size;
        const float* c_attn_w = w; w += 3 * n_embd * n_embd;
        const float* c_attn_b = w; w += 3 * n_embd;
        const float* c_proj_w = w; w += n_embd * n_embd;
        const float* c_proj_b = w; w += n_embd;
        const float* ln_2_w = w; w += n_embd;
        const float* ln_2_b = w; w += n_embd;
        const float* mlp_fc_w = w; w += 4 * n_embd * n_embd;
        const float* mlp_fc_b = w; w += 4 * n_embd;
        const float* mlp_proj_w = w; w += n_embd * 4 * n_embd;
        const float* mlp_proj_b = w; w += n_embd;
        
        // ln_1
        layer_norm(engine->x_buf, ln_1_w, ln_1_b, engine->norm_buf, n_embd);
        
        // Q, K, V
        float qkv[3 * MAX_EMBD];
        matvec_fp32(c_attn_w, engine->norm_buf, qkv, 3 * n_embd, n_embd);
        for (uint32_t i = 0; i < 3 * n_embd; i++) qkv[i] += c_attn_b[i];
        
        float *q = qkv;
        float *k = qkv + n_embd;
        float *v = qkv + 2 * n_embd;
        
        // Cache K, V
        for (uint32_t i = 0; i < n_embd; i++) {
            engine->kv_cache[l][0][pos][i] = k[i];
            engine->kv_cache[l][1][pos][i] = v[i];
        }
        
        // Attention
        uint32_t head_size = n_embd / n_head;
        float att_out[MAX_EMBD];
        for (uint32_t h = 0; h < n_head; h++) {
            float *q_h = q + h * head_size;
            
            for (int t = 0; t <= pos; t++) {
                float *k_t = engine->kv_cache[l][0][t] + h * head_size;
                float score = 0.0f;
                for (uint32_t i = 0; i < head_size; i++) {
                    score += q_h[i] * k_t[i];
                }
                engine->attn_scores[t] = score / sqrtf((float)head_size);
            }
            
            float max_val = engine->attn_scores[0];
            for (int t = 1; t <= pos; t++) {
                if (engine->attn_scores[t] > max_val) max_val = engine->attn_scores[t];
            }
            float sum = 0.0f;
            for (int t = 0; t <= pos; t++) {
                engine->attn_scores[t] = expf(engine->attn_scores[t] - max_val);
                sum += engine->attn_scores[t];
            }
            for (int t = 0; t <= pos; t++) {
                engine->attn_scores[t] /= sum;
            }
            
            float *out_h = att_out + h * head_size;
            for (uint32_t i = 0; i < head_size; i++) out_h[i] = 0.0f;
            
            for (int t = 0; t <= pos; t++) {
                float *v_t = engine->kv_cache[l][1][t] + h * head_size;
                float p = engine->attn_scores[t];
                for (uint32_t i = 0; i < head_size; i++) {
                    out_h[i] += p * v_t[i];
                }
            }
        }
        
        // Attention projection & residual
        matvec_fp32(c_proj_w, att_out, engine->q_buf, n_embd, n_embd);
        for (uint32_t i = 0; i < n_embd; i++) {
            engine->x_buf[i] += engine->q_buf[i] + c_proj_b[i];
        }
        
        // ln_2
        layer_norm(engine->x_buf, ln_2_w, ln_2_b, engine->norm_buf, n_embd);
        
        // MLP & residual
        matvec_fp32(mlp_fc_w, engine->norm_buf, engine->mlp_hidden, 4 * n_embd, n_embd);
        for (uint32_t i = 0; i < 4 * n_embd; i++) {
            engine->mlp_hidden[i] = gelu(engine->mlp_hidden[i] + mlp_fc_b[i]);
        }
        
        matvec_fp32(mlp_proj_w, engine->mlp_hidden, engine->q_buf, n_embd, 4 * n_embd);
        for (uint32_t i = 0; i < n_embd; i++) {
            engine->x_buf[i] += engine->q_buf[i] + mlp_proj_b[i];
        }
    }
    
    // Final ln_f
    const float* ln_f_w = w; w += n_embd;
    const float* ln_f_b = w; w += n_embd;
    layer_norm(engine->x_buf, ln_f_w, ln_f_b, engine->norm_buf, n_embd);
    
    // LM Head (weight tying with wte)
    matvec_fp32(wte, engine->norm_buf, out_logits, vocab_size, n_embd);
}
