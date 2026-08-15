#include "esp32_llm_engine.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>

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

// Quantize float array to INT4 [-8, 7] and calculate scale
void quantize_int4(const float* x, uint8_t* out, float* out_scale, int dim) {
    float max_abs = 0.0f;
    for (int i=0; i<dim; i++) {
        if (fabsf(x[i]) > max_abs) max_abs = fabsf(x[i]);
    }
    float scale = max_abs / 7.0f;
    if (scale == 0) scale = 1.0f;
    *out_scale = scale;
    float inv_scale = 1.0f / scale;
    
    for (int i=0; i<dim; i+=2) {
        int v0 = (int)roundf(x[i] * inv_scale);
        if (v0 < -8) v0 = -8; if (v0 > 7) v0 = 7;
        
        int v1 = 0;
        if (i+1 < dim) {
            v1 = (int)roundf(x[i+1] * inv_scale);
            if (v1 < -8) v1 = -8; if (v1 > 7) v1 = 7;
        }
        
        out[i/2] = (uint8_t)(((v1 & 0x0F) << 4) | (v0 & 0x0F));
    }
}

// Dequantize INT4 to float
void dequantize_int4(const uint8_t* in, float scale, float* out, int dim) {
    for (int i=0; i<dim; i+=2) {
        uint8_t b = in[i/2];
        int8_t v0 = (int8_t)(b & 0x0F); if (v0 & 8) v0 |= 0xF0; // sign extend
        int8_t v1 = (int8_t)((b >> 4) & 0x0F); if (v1 & 8) v1 |= 0xF0;
        
        out[i] = v0 * scale;
        if (i+1 < dim) out[i+1] = v1 * scale;
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

void matvec_int4(const uint8_t *packed_w, const float *scales, const float *vec, float *out, int rows, int cols) {
    int bytes_per_row = (cols + 1) / 2;
    static float w_buf[MAX_EMBD];
    
    for (int r = 0; r < rows; r++) {
        const uint8_t *row_packed = packed_w + r * bytes_per_row;
        float scale = scales[r];
        dequantize_int4(row_packed, scale, w_buf, cols);
        
        float acc = 0.0f;
        for (int c = 0; c < cols; c++) {
            acc += w_buf[c] * vec[c];
        }
        out[r] = acc;
    }
}

// ─── Math & Matrix Operations (Ternary 1.58-bit) ───
// ESP-DSP friendly Ternary Math (-1, 0, 1 -> Addition and Subtraction only)
void matvec_ternary_2bit(const uint8_t *packed_w, float scale, const float *vec, float *out, int rows, int cols) {
    int bytes_per_row = (cols + 3) / 4;
    static int8_t w_buf[MAX_EMBD];
    
    for (int r = 0; r < rows; r++) {
        const uint8_t *row_packed = packed_w + r * bytes_per_row;
        unpack_2bit_block(row_packed, w_buf, cols);
        float acc = 0.0f;
        
        // This inner loop can be heavily optimized using ESP-DSP or custom assembly
        for (int c = 0; c < cols; c++) {
            if (w_buf[c] == 1) acc += vec[c];
            else if (w_buf[c] == -1) acc -= vec[c];
        }
        out[r] = acc * scale;
    }
}

void apply_rope(float* vec, int pos, int num_heads, int head_size) {
    int d = head_size / 2;
    for (int h = 0; h < num_heads; h++) {
        float* v = vec + h * head_size;
        for (int i = 0; i < d; i++) {
            float theta = powf(10000.0f, -((float)(2 * i)) / head_size);
            float m_theta = pos * theta;
            float cos_t = cosf(m_theta);
            float sin_t = sinf(m_theta);
            
            float v0 = v[i];
            float v1 = v[i + d];
            
            v[i]     = v0 * cos_t - v1 * sin_t;
            v[i + d] = v1 * cos_t + v0 * sin_t;
        }
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
    if (engine->config.n_head == 0) engine->config.n_head = 1;
    if (engine->config.n_kv_head == 0) engine->config.n_kv_head = engine->config.n_head;
    engine->model_weights_ptr = weights_ptr;
    esp32_llm_reset_kv_cache(engine);
}

void esp32_llm_reset_kv_cache(ESP32LLMEngine *engine) {
    memset(engine->kv_cache, 0, sizeof(engine->kv_cache));
}

void esp32_llm_forward_step(ESP32LLMEngine *engine, int token_id, int pos, float *out_logits) {
    uint32_t n_layer = engine->config.n_layer;
    uint32_t n_head = engine->config.n_head;
    uint32_t n_kv_head = engine->config.n_kv_head;
    uint32_t n_embd = engine->config.n_embd;
    uint32_t block_size = engine->config.block_size;
    uint32_t vocab_size = engine->config.vocab_size;
    WeightFormat fmt = engine->config.format;
    
    if (fmt != WEIGHT_FMT_TERNARY_2BIT && fmt != WEIGHT_FMT_INT4 && fmt != WEIGHT_FMT_FP32 && fmt != WEIGHT_FMT_TERNARY_1_58BIT) {
        printf("ERROR: Unsupported weight format %d in forward_step!\n", (int)fmt);
        return;
    }
    
    const uint8_t* ptr = engine->model_weights_ptr;
    
    // Embeddings (Stored in FP32)
    const float* wte = (const float*)ptr; ptr += vocab_size * n_embd * sizeof(float);
    
    for (uint32_t i = 0; i < n_embd; i++) {
        engine->x_buf[i] = wte[token_id * n_embd + i];
    }
    
    uint32_t head_size = n_embd / n_head;
    uint32_t num_kv_groups = n_head / n_kv_head;
    
    for (uint32_t l = 0; l < n_layer; l++) {
        const float* ln_1_w = (const float*)ptr; ptr += n_embd * sizeof(float);
        const float* ln_1_b = (const float*)ptr; ptr += n_embd * sizeof(float);
        
        layer_norm(engine->x_buf, ln_1_w, ln_1_b, engine->norm_buf, n_embd);
        
        if (fmt == WEIGHT_FMT_TERNARY_2BIT) {
            float q_scale = *(const float*)ptr; ptr += sizeof(float);
            const uint8_t* q_proj_w = ptr; ptr += ((n_head * head_size * n_embd) + 3) / 4;
            const float* q_proj_b = (const float*)ptr; ptr += n_head * head_size * sizeof(float);
            
            float k_scale = *(const float*)ptr; ptr += sizeof(float);
            const uint8_t* k_proj_w = ptr; ptr += ((n_kv_head * head_size * n_embd) + 3) / 4;
            const float* k_proj_b = (const float*)ptr; ptr += n_kv_head * head_size * sizeof(float);
            
            float v_scale = *(const float*)ptr; ptr += sizeof(float);
            const uint8_t* v_proj_w = ptr; ptr += ((n_kv_head * head_size * n_embd) + 3) / 4;
            const float* v_proj_b = (const float*)ptr; ptr += n_kv_head * head_size * sizeof(float);
            
            matvec_ternary_2bit(q_proj_w, q_scale, engine->norm_buf, engine->q_buf, n_head * head_size, n_embd);
            for (uint32_t i = 0; i < n_head * head_size; i++) engine->q_buf[i] += q_proj_b[i];
            
            matvec_ternary_2bit(k_proj_w, k_scale, engine->norm_buf, engine->k_buf, n_kv_head * head_size, n_embd);
            for (uint32_t i = 0; i < n_kv_head * head_size; i++) engine->k_buf[i] += k_proj_b[i];
            
            matvec_ternary_2bit(v_proj_w, v_scale, engine->norm_buf, engine->v_buf, n_kv_head * head_size, n_embd);
            for (uint32_t i = 0; i < n_kv_head * head_size; i++) engine->v_buf[i] += v_proj_b[i];
        } else if (fmt == WEIGHT_FMT_INT4) {
            const float* q_scales = (const float*)ptr; ptr += n_head * head_size * sizeof(float);
            const uint8_t* q_proj_w = ptr; ptr += ((n_head * head_size * n_embd) + 1) / 2;
            const float* q_proj_b = (const float*)ptr; ptr += n_head * head_size * sizeof(float);
            
            const float* k_scales = (const float*)ptr; ptr += n_kv_head * head_size * sizeof(float);
            const uint8_t* k_proj_w = ptr; ptr += ((n_kv_head * head_size * n_embd) + 1) / 2;
            const float* k_proj_b = (const float*)ptr; ptr += n_kv_head * head_size * sizeof(float);
            
            const float* v_scales = (const float*)ptr; ptr += n_kv_head * head_size * sizeof(float);
            const uint8_t* v_proj_w = ptr; ptr += ((n_kv_head * head_size * n_embd) + 1) / 2;
            const float* v_proj_b = (const float*)ptr; ptr += n_kv_head * head_size * sizeof(float);
            
            matvec_int4(q_proj_w, q_scales, engine->norm_buf, engine->q_buf, n_head * head_size, n_embd);
            for (uint32_t i = 0; i < n_head * head_size; i++) engine->q_buf[i] += q_proj_b[i];
            
            matvec_int4(k_proj_w, k_scales, engine->norm_buf, engine->k_buf, n_kv_head * head_size, n_embd);
            for (uint32_t i = 0; i < n_kv_head * head_size; i++) engine->k_buf[i] += k_proj_b[i];
            
            matvec_int4(v_proj_w, v_scales, engine->norm_buf, engine->v_buf, n_kv_head * head_size, n_embd);
            for (uint32_t i = 0; i < n_kv_head * head_size; i++) engine->v_buf[i] += v_proj_b[i];
        } else if (fmt == WEIGHT_FMT_FP32) {
            const float* q_proj_w = (const float*)ptr; ptr += n_head * head_size * n_embd * sizeof(float);
            const float* q_proj_b = (const float*)ptr; ptr += n_head * head_size * sizeof(float);
            
            const float* k_proj_w = (const float*)ptr; ptr += n_kv_head * head_size * n_embd * sizeof(float);
            const float* k_proj_b = (const float*)ptr; ptr += n_kv_head * head_size * sizeof(float);
            
            const float* v_proj_w = (const float*)ptr; ptr += n_kv_head * head_size * n_embd * sizeof(float);
            const float* v_proj_b = (const float*)ptr; ptr += n_kv_head * head_size * sizeof(float);
            
            matvec_fp32(q_proj_w, engine->norm_buf, engine->q_buf, n_head * head_size, n_embd);
            for (uint32_t i = 0; i < n_head * head_size; i++) engine->q_buf[i] += q_proj_b[i];
            
            matvec_fp32(k_proj_w, engine->norm_buf, engine->k_buf, n_kv_head * head_size, n_embd);
            for (uint32_t i = 0; i < n_kv_head * head_size; i++) engine->k_buf[i] += k_proj_b[i];
            
            matvec_fp32(v_proj_w, engine->norm_buf, engine->v_buf, n_kv_head * head_size, n_embd);
            for (uint32_t i = 0; i < n_kv_head * head_size; i++) engine->v_buf[i] += v_proj_b[i];
        }
        
        apply_rope(engine->q_buf, pos, n_head, head_size);
        apply_rope(engine->k_buf, pos, n_kv_head, head_size);
        
        for (uint32_t kvh = 0; kvh < n_kv_head; kvh++) {
            float* k_head = engine->k_buf + kvh * head_size;
            float* v_head = engine->v_buf + kvh * head_size;
            quantize_int4(k_head, engine->kv_cache[l][0][pos][kvh], &engine->kv_cache_scales[l][0][pos][kvh], head_size);
            quantize_int4(v_head, engine->kv_cache[l][1][pos][kvh], &engine->kv_cache_scales[l][1][pos][kvh], head_size);
        }
        
        static float att_out[MAX_EMBD];
        for (uint32_t h = 0; h < n_head; h++) {
            uint32_t kvh = h / num_kv_groups;
            float *q_h = engine->q_buf + h * head_size;
            
            for (int t = 0; t <= pos; t++) {
                float k_t[MAX_EMBD / MAX_HEADS];
                dequantize_int4(engine->kv_cache[l][0][t][kvh], engine->kv_cache_scales[l][0][t][kvh], k_t, head_size);
                float score = 0.0f;
                for (uint32_t i = 0; i < head_size; i++) score += q_h[i] * k_t[i];
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
            for (int t = 0; t <= pos; t++) engine->attn_scores[t] /= sum;
            
            float *out_h = att_out + h * head_size;
            for (uint32_t i = 0; i < head_size; i++) out_h[i] = 0.0f;
            
            for (int t = 0; t <= pos; t++) {
                float v_t[MAX_EMBD / MAX_HEADS];
                dequantize_int4(engine->kv_cache[l][1][t][kvh], engine->kv_cache_scales[l][1][t][kvh], v_t, head_size);
                float p = engine->attn_scores[t];
                for (uint32_t i = 0; i < head_size; i++) out_h[i] += p * v_t[i];
            }
        }
        
        if (fmt == WEIGHT_FMT_TERNARY_2BIT) {
            float c_proj_scale = *(const float*)ptr; ptr += sizeof(float);
            const uint8_t* c_proj_w = ptr; ptr += ((n_embd * n_embd) + 3) / 4;
            const float* c_proj_b = (const float*)ptr; ptr += n_embd * sizeof(float);
            
            matvec_ternary_2bit(c_proj_w, c_proj_scale, att_out, engine->q_buf, n_embd, n_embd);
            for (uint32_t i = 0; i < n_embd; i++) engine->x_buf[i] += engine->q_buf[i] + c_proj_b[i];
        } else if (fmt == WEIGHT_FMT_INT4) {
            const float* c_scales = (const float*)ptr; ptr += n_embd * sizeof(float);
            const uint8_t* c_proj_w = ptr; ptr += ((n_embd * n_embd) + 1) / 2;
            const float* c_proj_b = (const float*)ptr; ptr += n_embd * sizeof(float);
            
            matvec_int4(c_proj_w, c_scales, att_out, engine->q_buf, n_embd, n_embd);
            for (uint32_t i = 0; i < n_embd; i++) engine->x_buf[i] += engine->q_buf[i] + c_proj_b[i];
        } else if (fmt == WEIGHT_FMT_FP32) {
            const float* c_proj_w = (const float*)ptr; ptr += n_embd * n_embd * sizeof(float);
            const float* c_proj_b = (const float*)ptr; ptr += n_embd * sizeof(float);
            
            matvec_fp32(c_proj_w, att_out, engine->q_buf, n_embd, n_embd);
            for (uint32_t i = 0; i < n_embd; i++) engine->x_buf[i] += engine->q_buf[i] + c_proj_b[i];
        }
        
        const float* ln_2_w = (const float*)ptr; ptr += n_embd * sizeof(float);
        const float* ln_2_b = (const float*)ptr; ptr += n_embd * sizeof(float);
        layer_norm(engine->x_buf, ln_2_w, ln_2_b, engine->norm_buf, n_embd);
        
        if (fmt == WEIGHT_FMT_TERNARY_2BIT) {
            float mlp_fc_scale = *(const float*)ptr; ptr += sizeof(float);
            const uint8_t* mlp_fc_w = ptr; ptr += ((4 * n_embd * n_embd) + 3) / 4;
            const float* mlp_fc_b = (const float*)ptr; ptr += 4 * n_embd * sizeof(float);
            
            matvec_ternary_2bit(mlp_fc_w, mlp_fc_scale, engine->norm_buf, engine->mlp_hidden, 4 * n_embd, n_embd);
            for (uint32_t i = 0; i < 4 * n_embd; i++) engine->mlp_hidden[i] = gelu(engine->mlp_hidden[i] + mlp_fc_b[i]);
            
            float mlp_proj_scale = *(const float*)ptr; ptr += sizeof(float);
            const uint8_t* mlp_proj_w = ptr; ptr += ((n_embd * 4 * n_embd) + 3) / 4;
            const float* mlp_proj_b = (const float*)ptr; ptr += n_embd * sizeof(float);
            
            matvec_ternary_2bit(mlp_proj_w, mlp_proj_scale, engine->mlp_hidden, engine->q_buf, n_embd, 4 * n_embd);
            for (uint32_t i = 0; i < n_embd; i++) engine->x_buf[i] += engine->q_buf[i] + mlp_proj_b[i];
        } else if (fmt == WEIGHT_FMT_INT4) {
            const float* mlp_fc_scales = (const float*)ptr; ptr += 4 * n_embd * sizeof(float);
            const uint8_t* mlp_fc_w = ptr; ptr += ((4 * n_embd * n_embd) + 1) / 2;
            const float* mlp_fc_b = (const float*)ptr; ptr += 4 * n_embd * sizeof(float);
            
            matvec_int4(mlp_fc_w, mlp_fc_scales, engine->norm_buf, engine->mlp_hidden, 4 * n_embd, n_embd);
            for (uint32_t i = 0; i < 4 * n_embd; i++) engine->mlp_hidden[i] = gelu(engine->mlp_hidden[i] + mlp_fc_b[i]);
            
            const float* mlp_proj_scales = (const float*)ptr; ptr += n_embd * sizeof(float);
            const uint8_t* mlp_proj_w = ptr; ptr += ((n_embd * 4 * n_embd) + 1) / 2;
            const float* mlp_proj_b = (const float*)ptr; ptr += n_embd * sizeof(float);
            
            matvec_int4(mlp_proj_w, mlp_proj_scales, engine->mlp_hidden, engine->q_buf, n_embd, 4 * n_embd);
            for (uint32_t i = 0; i < n_embd; i++) engine->x_buf[i] += engine->q_buf[i] + mlp_proj_b[i];
        } else if (fmt == WEIGHT_FMT_FP32) {
            const float* mlp_fc_w = (const float*)ptr; ptr += 4 * n_embd * n_embd * sizeof(float);
            const float* mlp_fc_b = (const float*)ptr; ptr += 4 * n_embd * sizeof(float);
            
            matvec_fp32(mlp_fc_w, engine->norm_buf, engine->mlp_hidden, 4 * n_embd, n_embd);
            for (uint32_t i = 0; i < 4 * n_embd; i++) engine->mlp_hidden[i] = gelu(engine->mlp_hidden[i] + mlp_fc_b[i]);
            
            const float* mlp_proj_w = (const float*)ptr; ptr += n_embd * 4 * n_embd * sizeof(float);
            const float* mlp_proj_b = (const float*)ptr; ptr += n_embd * sizeof(float);
            
            matvec_fp32(mlp_proj_w, engine->mlp_hidden, engine->q_buf, n_embd, 4 * n_embd);
            for (uint32_t i = 0; i < n_embd; i++) engine->x_buf[i] += engine->q_buf[i] + mlp_proj_b[i];
        }
        
        vTaskDelay(1);
    }
    
    const float* ln_f_w = (const float*)ptr; ptr += n_embd * sizeof(float);
    const float* ln_f_b = (const float*)ptr; ptr += n_embd * sizeof(float);
    layer_norm(engine->x_buf, ln_f_w, ln_f_b, engine->norm_buf, n_embd);
    
    matvec_fp32(wte, engine->norm_buf, out_logits, vocab_size, n_embd);
}
