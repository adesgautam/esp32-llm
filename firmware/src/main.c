#include <stdio.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_partition.h"
#include "esp_log.h"
#include "esp_spi_flash.h"
#include "esp_timer.h"
#include "model_runner.h"
#include "esp32_llm_engine.h"
#include "bpe_vocab.h"

static const char *TAG = "MODEL_RUNNER";

static ESP32LLMEngine s_engine;
static const void *s_mapped_model_ptr = NULL;
static spi_flash_mmap_handle_t s_mmap_handle;

esp_err_t init_model_runner(void)
{
    ESP_LOGI(TAG, "Searching for 'model' partition in SPI Flash...");
    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, "model");
    if (!part) {
        ESP_LOGE(TAG, "Failed to find 'model' partition!");
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "Found partition '%s' at offset 0x%08x, size %d bytes", part->label, (unsigned int)part->address, (int)part->size);

    ESP_LOGI(TAG, "Mapping model partition into MMU memory space (XIP)...");
    esp_err_t err = esp_partition_mmap(part, 0, part->size, SPI_FLASH_MMAP_DATA, &s_mapped_model_ptr, &s_mmap_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "esp_partition_mmap failed: %d", err);
        return err;
    }

    ESP_LOGI(TAG, "Model partition successfully mapped at address %p", s_mapped_model_ptr);

    // Read config from binary header
    const uint32_t *hdr = (const uint32_t *)s_mapped_model_ptr;
    uint32_t magic = hdr[0];
    uint32_t version = hdr[1];
    
    ESP_LOGI(TAG, "Model header: magic=0x%08x version=%d", (unsigned)magic, (int)version);
    
    ESP32LLMConfigC config = {
        .n_layer = hdr[2],
        .n_head = hdr[3],
        .n_embd = hdr[4],
        .block_size = hdr[5],
        .vocab_size = hdr[6],
        .format = WEIGHT_FMT_FP32
    };

    ESP_LOGI(TAG, "Config: %dL/%dH/%dD, block=%d, vocab=%d",
             (int)config.n_layer, (int)config.n_head, (int)config.n_embd,
             (int)config.block_size, (int)config.vocab_size);

    esp32_llm_init(&s_engine, &config, (const uint8_t *)s_mapped_model_ptr);
    ESP_LOGI(TAG, "ESP32LLM BPE engine initialized.");
    return ESP_OK;
}

// ─── BPE Encoding (user text → token IDs) ───

static int bpe_encode_prompt(const char *text, int *out_tokens, int max_tokens) {
    // Step 1: Character-level base encoding
    int n = 0;
    for (int i = 0; text[i] && n < max_tokens; i++) {
        char c = text[i];
        if (c >= 'A' && c <= 'Z') c += 32; // Lowercase
        
        // Find in base vocabulary (first BPE_N_BASE tokens are single chars)
        int found = -1;
        for (int j = 0; j < BPE_N_BASE; j++) {
            if (bpe_vocab[j][0] == c && bpe_vocab[j][1] == '\0') {
                found = j;
                break;
            }
        }
        if (found >= 0) {
            out_tokens[n++] = found;
        }
    }
    
    // Step 2: Apply BPE merges in order
    for (int m = 0; m < BPE_N_MERGES; m++) {
        int a = bpe_merges[m][0];
        int b = bpe_merges[m][1];
        int new_id = BPE_N_BASE + m;
        
        int new_n = 0;
        int i = 0;
        while (i < n) {
            if (i < n - 1 && out_tokens[i] == a && out_tokens[i+1] == b) {
                out_tokens[new_n++] = new_id;
                i += 2;
            } else {
                out_tokens[new_n++] = out_tokens[i];
                i++;
            }
        }
        n = new_n;
    }
    
    return n;
}

// ─── Sampling ───

static uint32_t s_rng_state = 0xDEADBEEF;

static uint32_t xorshift32(void) {
    uint32_t x = s_rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    s_rng_state = x;
    return x;
}

static float rand_float(void) {
    return (float)(xorshift32() & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

static int sample_top_k(float* logits, int vocab_size, float temperature, int top_k) {
    // Temperature scaling
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= temperature;
    }
    
    // Find top-k indices
    int indices[MAX_VOCAB_SIZE];
    for (int i = 0; i < vocab_size; i++) indices[i] = i;
    
    for (int i = 0; i < top_k && i < vocab_size; i++) {
        for (int j = i + 1; j < vocab_size; j++) {
            if (logits[indices[j]] > logits[indices[i]]) {
                int tmp = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp;
            }
        }
    }
    
    // Softmax over top-k
    float max_val = logits[indices[0]];
    float sum = 0.0f;
    float probs[MAX_VOCAB_SIZE];
    
    for (int i = 0; i < top_k && i < vocab_size; i++) {
        probs[i] = expf(logits[indices[i]] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < top_k && i < vocab_size; i++) {
        probs[i] /= sum;
    }
    
    float r = rand_float();
    float cumulative = 0.0f;
    for (int i = 0; i < top_k && i < vocab_size; i++) {
        cumulative += probs[i];
        if (r <= cumulative) return indices[i];
    }
    return indices[0];
}

// ─── Generation ───

void run_poetry_generation(const char *prompt, int max_tokens)
{
    if (!s_mapped_model_ptr) {
        ESP_LOGE(TAG, "Model runner not initialized!");
        return;
    }

    s_rng_state = (uint32_t)esp_timer_get_time();

    uint32_t vocab_size = s_engine.config.vocab_size;
    uint32_t block_size = s_engine.config.block_size;

    // BPE-encode the prompt
    int prompt_tokens[MAX_BLOCK_SIZE];
    int prompt_len = bpe_encode_prompt(prompt, prompt_tokens, block_size);
    
    ESP_LOGI(TAG, "BPE prompt: '%s' -> %d tokens", prompt, prompt_len);
    esp32_llm_reset_kv_cache(&s_engine);

    float logits[MAX_VOCAB_SIZE];

    for (int step = 0; step < max_tokens; step++) {
        int token_id;
        
        if (step < prompt_len) {
            token_id = prompt_tokens[step];
        } else {
            token_id = sample_top_k(logits, vocab_size, 0.8f, 8);
            // BPE decode: print the token string
            if (token_id >= 0 && token_id < (int)vocab_size && token_id < BPE_VOCAB_SIZE) {
                printf("%s", bpe_vocab[token_id]);
                fflush(stdout);
            }
        }

        esp32_llm_forward_step(&s_engine, token_id, step % block_size, logits);
        
        vTaskDelay(pdMS_TO_TICKS(5));
    }

    ESP_LOGI(TAG, "Generation complete.");
}
