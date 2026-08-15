#include <Arduino.h>
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
#include "esp_task_wdt.h"
#include "esp_rom_crc.h"
#include "web_ui.h"
extern "C" {
#include "esp32_llm_engine.h"
#include "bpe_vocab.h"
}

static const char *TAG = "MODEL_RUNNER";

static ESP32LLMEngine s_engine;
static const void *s_mapped_model_ptr = NULL;
static spi_flash_mmap_handle_t s_mmap_handle;

esp_err_t init_model_runner(void)
{
    printf("Searching for 'model' partition in SPI Flash...\n");
    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)0x40, "model");
    if (!part) {
        printf("ERROR: Failed to find 'model' partition!\n");
        return ESP_FAIL;
    }

    printf("Found partition '%s' at offset 0x%08x, size %d bytes\n", part->label, (unsigned int)part->address, (int)part->size);

    printf("Mapping model partition into MMU memory space (XIP)...\n");
    esp_err_t err = esp_partition_mmap(part, 0, part->size, SPI_FLASH_MMAP_DATA, &s_mapped_model_ptr, &s_mmap_handle);
    if (err != ESP_OK) {
        printf("ERROR: esp_partition_mmap failed: %d\n", err);
        return err;
    }

    printf("Model partition successfully mapped at address %p\n", s_mapped_model_ptr);

    // Read config from binary header
    const uint32_t *hdr = (const uint32_t *)s_mapped_model_ptr;
    uint32_t magic = hdr[0];
    uint32_t version = hdr[1];
    
    printf("Model header: magic=0x%08x version=%d\n", (unsigned)magic, (int)version);
    
    ESP32LLMConfigC config = {0};
    uint32_t header_bytes = 0;
    
    if (magic == 0x54504f45) { // 'TPOE' (Ternary Version 3)
        config.format = WEIGHT_FMT_TERNARY_2BIT;
        config.n_layer = hdr[2];
        config.n_head = hdr[3];
        config.n_kv_head = (hdr[4] == 0) ? hdr[3] : hdr[4];
        config.n_embd = hdr[5];
        config.block_size = (hdr[6] == 0 || hdr[6] > MAX_BLOCK_SIZE) ? MAX_BLOCK_SIZE : hdr[6];
        uint32_t expected_crc = hdr[8];
        uint32_t payload_size = hdr[9];
        header_bytes = 40;
        printf("Detected Format: Ternary 1.58-bit (Version %d), Checksum: 0x%08x\n", (int)version, (unsigned)expected_crc);
        
        printf("Verifying model checksum...\n");
        if (part->size < header_bytes + payload_size) {
            printf("WARNING: Partition size %d is smaller than required %d bytes!\n", (int)part->size, (int)(header_bytes + payload_size));
        } else {
            uint32_t calc_crc = esp_rom_crc32_le(0, (const uint8_t*)s_mapped_model_ptr + header_bytes, payload_size);
            if (calc_crc != expected_crc) {
                printf("WARNING: Checksum MISMATCH! Expected 0x%08x, got 0x%08x\n", (unsigned)expected_crc, (unsigned)calc_crc);
            } else {
                printf("Checksum OK.\n");
            }
        }
    } else if (magic == 0x54504934) { // 'TPI4' (INT4 Version 1)
        config.format = WEIGHT_FMT_INT4;
        config.n_layer = hdr[2];
        config.n_head = hdr[3];
        config.n_kv_head = hdr[3]; // Old models didn't have MQA
        config.n_embd = hdr[4];
        config.block_size = (hdr[5] == 0 || hdr[5] > MAX_BLOCK_SIZE) ? MAX_BLOCK_SIZE : hdr[5];
        config.vocab_size = hdr[6];
        header_bytes = 28;
        printf("Detected Format: INT4 (Version %d)\n", (int)version);
    } else if (magic == 0x54504632) { // 'TPF2' (FP32 Version 2)
        config.format = WEIGHT_FMT_FP32;
        config.n_layer = hdr[2];
        config.n_head = hdr[3];
        config.n_kv_head = hdr[3];
        config.n_embd = hdr[4];
        config.block_size = (hdr[5] == 0 || hdr[5] > MAX_BLOCK_SIZE) ? MAX_BLOCK_SIZE : hdr[5];
        config.vocab_size = hdr[6];
        header_bytes = 28;
        printf("Detected Format: FP32 (Version %d)\n", (int)version);
    } else {
        printf("ERROR: Unknown magic number 0x%08x\n", (unsigned)magic);
        return ESP_FAIL;
    }

    printf("Config: %dL/%dH(kv:%d)/%dD, block=%d, vocab=%d\n",
             (int)config.n_layer, (int)config.n_head, (int)config.n_kv_head, (int)config.n_embd,
             (int)config.block_size, (int)config.vocab_size);

    esp32_llm_init(&s_engine, &config, (const uint8_t *)s_mapped_model_ptr + header_bytes);
    printf("ESP32LLM BPE engine initialized.\n");
    return ESP_OK;
}

// ─── BPE Encoding (user text → token IDs) ───

static int bpe_encode_prompt(const char *text, int *out_tokens, int max_tokens) {
    // Step 1: Character-level base encoding
    int n = 0;
    for (int i = 0; text[i] && n < max_tokens; i++) {
        char c = text[i];
        // Find in base vocabulary (first BPE_N_BASE tokens are single chars)
        int found = -1;
        int fallback_id = 0; // default to 0 (space usually)
        for (int j = 0; j < BPE_N_BASE; j++) {
            if (bpe_vocab[j][0] == c && bpe_vocab[j][1] == '\0') {
                found = j;
            }
            if (bpe_vocab[j][0] == '?' && bpe_vocab[j][1] == '\0') {
                fallback_id = j;
            }
        }
        
        if (found >= 0) {
            out_tokens[n++] = found;
        } else {
            out_tokens[n++] = fallback_id;
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

static int s_indices[MAX_VOCAB_SIZE];
static float s_probs[MAX_VOCAB_SIZE];

static int sample_top_k(float* logits, int vocab_size, float temperature, int top_k) {
    // Temperature scaling
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= temperature;
    }
    
    // Find top-k indices
    for (int i = 0; i < vocab_size; i++) s_indices[i] = i;
    
    for (int i = 0; i < top_k && i < vocab_size; i++) {
        for (int j = i + 1; j < vocab_size; j++) {
            if (logits[s_indices[j]] > logits[s_indices[i]]) {
                int tmp = s_indices[i];
                s_indices[i] = s_indices[j];
                s_indices[j] = tmp;
            }
        }
    }
    
    // Softmax over top-k
    float max_val = logits[s_indices[0]];
    float sum = 0.0f;
    
    for (int i = 0; i < top_k && i < vocab_size; i++) {
        s_probs[i] = expf(logits[s_indices[i]] - max_val);
        sum += s_probs[i];
    }
    for (int i = 0; i < top_k && i < vocab_size; i++) {
        s_probs[i] /= sum;
    }
    
    float r = rand_float();
    float cumulative = 0.0f;
    for (int i = 0; i < top_k && i < vocab_size; i++) {
        cumulative += s_probs[i];
        if (r <= cumulative) return s_indices[i];
    }
    return s_indices[0];
}

// ─── Generation ───

const char* get_model_name(const ESP32LLMConfigC* config) {
    if (config->n_layer == 4 && config->n_head == 4 && config->n_embd == 64) return "Micro-LM-Pico (Ternary)";
    if (config->n_layer == 4 && config->n_head == 4 && config->n_embd == 256) return "Micro-LM-Pro (Ternary)";
    if (config->n_layer == 4 && config->n_head == 8 && config->n_embd == 512) return "Micro-LM-Ultra (Ternary)";
    if (config->n_layer == 2 && config->n_head == 2 && config->n_embd == 48) return "TinyPoet-Nano (Ternary)";
    return "Unknown ESP32-LLM Variant";
}

static float g_temperature = 0.8f;
static int g_top_k = 8;
static int g_max_tokens = 100;

void run_poetry_generation(const char *prompt, int max_tokens)
{
    if (!s_mapped_model_ptr) {
        printf("ERROR: Model runner not initialized!\n");
        return;
    }

    s_rng_state = (uint32_t)esp_timer_get_time();

    uint32_t vocab_size = s_engine.config.vocab_size;
    uint32_t block_size = s_engine.config.block_size;

    // BPE-encode the prompt (static to avoid stack overflow)
    static int prompt_tokens[MAX_BLOCK_SIZE];
    int prompt_len = bpe_encode_prompt(prompt, prompt_tokens, block_size);
    
    printf("BPE prompt: '%s' -> %d tokens\n", prompt, prompt_len);
    esp32_llm_reset_kv_cache(&s_engine);

    static float logits[MAX_VOCAB_SIZE];

    int gen_count = 0;
    int64_t start_time = esp_timer_get_time();

    for (int step = 0; step < max_tokens; step++) {
        int token_id;
        
        if (step < prompt_len) {
            token_id = prompt_tokens[step];
        } else {
            token_id = sample_top_k(logits, vocab_size, g_temperature, g_top_k);
            gen_count++;
            // BPE decode: print the token string
            if (token_id >= 0 && token_id < (int)vocab_size && token_id < BPE_VOCAB_SIZE) {
                printf("%s", bpe_vocab[token_id]);
                fflush(stdout);
                send_token_to_web_clients(bpe_vocab[token_id]);
            }
        }

        esp32_llm_forward_step(&s_engine, token_id, step % block_size, logits);
    }

    int64_t end_time = esp_timer_get_time();
    double total_sec = (double)(end_time - start_time) / 1000000.0;
    double tps = gen_count > 0 ? (double)gen_count / total_sec : 0.0;

    printf("\n");
    printf("==========================================\n");
    printf(" Inference Stats: %s\n", get_model_name(&s_engine.config));
    printf(" Prompt tokens: %d\n", prompt_len);
    printf(" Generated tokens: %d\n", gen_count);
    printf(" Total time: %.3f sec\n", total_sec);
    printf(" Performance: %.2f tokens/sec (TPS)\n", tps);
    printf("==========================================\n");
    
    send_token_to_web_clients(nullptr); // Signal DONE
}

void setup() {
    Serial.begin(115200);
    // Disable task watchdog entirely — LLM inference monopolizes CPU for extended periods
    esp_task_wdt_deinit();
    
    printf("\nESP32-LLM Starting...\n");
    if (init_model_runner() == ESP_OK) {
        printf("Model initialized.\n");
        init_web_ui();
        printf("\nReady! Connect to MicroLM-AP and visit 192.168.4.1\nOr type a prompt via Serial Monitor and press Enter:\n> ");
    }
}

static char input_buf[128];
static int input_len = 0;

void loop() {
    handle_web_ui_client();
    
    if (s_mapped_model_ptr) {
        if (is_generating_for_web()) {
            printf("\n[Web UI Prompt]: %s\n", get_web_prompt());
            run_poetry_generation(get_web_prompt(), g_max_tokens);
            clear_web_prompt();
            printf("\n> ");
        }

        if (Serial.available()) {
            int c = Serial.read();
            if (c == '\n' || c == '\r') {
                if (input_len > 0) {
                    input_buf[input_len] = '\0';
                    printf("\n");
                    
                    if (strncmp(input_buf, "/temp ", 6) == 0) {
                        g_temperature = atof(input_buf + 6);
                        printf("Temperature set to %.2f\n", g_temperature);
                    } else if (strncmp(input_buf, "/topk ", 6) == 0) {
                        g_top_k = atoi(input_buf + 6);
                        printf("Top-K set to %d\n", g_top_k);
                    } else if (strncmp(input_buf, "/len ", 5) == 0) {
                        g_max_tokens = atoi(input_buf + 5);
                        printf("Max length set to %d\n", g_max_tokens);
                    } else {
                        run_poetry_generation(input_buf, g_max_tokens);
                    }
                    
                    input_len = 0;
                    printf("\n> ");
                }
            } else if (input_len < sizeof(input_buf) - 1) {
                input_buf[input_len++] = (char)c;
                Serial.write(c); // Echo character back to terminal
            }
        }
    }
    vTaskDelay(10 / portTICK_PERIOD_MS);
}
