/*
 * Chapter 25 - the mini-LLM, running in a single pure-C file. No PyTorch, no
 * libraries beyond libm. This is the course's "three languages" payoff: the
 * transformer you trained in Chapter 24 is, at inference time, a few hundred
 * lines of loops over a file of numbers.
 *
 * It reads the .bin exported by export_llm_for_c.py (float32 or int8), runs
 * the full forward pass - token+position embeddings, N transformer blocks
 * (multi-head causal attention, MLP, layer norm, residuals), tied output head
 * - and generates text with temperature + top-k sampling (Chapter 23). The
 * BPE tokenizer (Chapter 20) is built in so it takes a text prompt.
 *
 * Before the first run (from the repository root):
 *     .venv/bin/python chapters/24-train-your-mini-llm/python/prepare_data.py   # if not done
 *     .venv/bin/python chapters/24-train-your-mini-llm/python/train_mini_llm.py --size small
 *     .venv/bin/python chapters/25-llm-inference-in-c/python/export_llm_for_c.py --size small
 *
 * Build and run from the repository root:
 *     make -C chapters/25-llm-inference-in-c/c
 *     ./chapters/25-llm-inference-in-c/c/build/llm_inference checkpoints/mini_llm_small.bin "The night was "
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EXPORT_MAGIC 0x4C4C4D31

/* ------------------------------------------------------------- model types */

/* A weight matrix, stored either as float32 or as int8 + a scale. read_matrix
 * hides which; matvec below dequantizes on the fly. */
typedef struct {
    int is_quantized;
    int row_count;
    int column_count;
    float *float_values;   /* used when is_quantized == 0 */
    int8_t *int8_values;   /* used when is_quantized == 1 */
    float scale;           /* dequantization factor for int8 */
} Matrix;

typedef struct {
    float *attention_norm_weight, *attention_norm_bias;
    Matrix qkv_weight;
    float *qkv_bias;
    Matrix attention_output_weight;
    float *attention_output_bias;
    float *mlp_norm_weight, *mlp_norm_bias;
    Matrix mlp_up_weight;
    float *mlp_up_bias;
    Matrix mlp_down_weight;
    float *mlp_down_bias;
} TransformerBlock;

typedef struct {
    int vocabulary_size, context_length, embedding_size, block_count, head_count;
    int is_quantized;
    float *token_embedding;      /* vocabulary_size x embedding_size (also the tied output head) */
    float *position_embedding;   /* context_length x embedding_size */
    TransformerBlock *blocks;
    float *final_norm_weight, *final_norm_bias;
} Model;

/* ------------------------------------------------------- file reading */

static float *read_floats(FILE *file, long count) {
    float *buffer = malloc(count * sizeof(float));
    if (buffer == NULL || fread(buffer, sizeof(float), count, file) != (size_t)count) {
        fprintf(stderr, "unexpected end of model file\n");
        exit(1);
    }
    return buffer;
}

/*
 * Read one weight matrix, honoring the file's quantization flag. Quantized
 * matrices arrive as a float32 scale followed by row_count*column_count int8s.
 */
static Matrix read_matrix(FILE *file, int row_count, int column_count, int is_quantized) {
    Matrix matrix = {is_quantized, row_count, column_count, NULL, NULL, 1.0f};
    long element_count = (long)row_count * column_count;
    if (is_quantized) {
        if (fread(&matrix.scale, sizeof(float), 1, file) != 1) {
            fprintf(stderr, "bad quantized matrix\n");
            exit(1);
        }
        matrix.int8_values = malloc(element_count);
        if (matrix.int8_values == NULL || fread(matrix.int8_values, 1, element_count, file) != (size_t)element_count) {
            fprintf(stderr, "bad quantized matrix data\n");
            exit(1);
        }
    } else {
        matrix.float_values = read_floats(file, element_count);
    }
    return matrix;
}

/*
 * Load the whole model in the exact order export_llm_for_c.py wrote it.
 */
static void load_model(const char *path, Model *model) {
    FILE *file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "cannot open %s (run the export script - see the top of this file)\n", path);
        exit(1);
    }
    int header[7];
    if (fread(header, sizeof(int), 7, file) != 7 || header[0] != EXPORT_MAGIC) {
        fprintf(stderr, "not a mini-LLM export file\n");
        exit(1);
    }
    model->vocabulary_size = header[1];
    model->context_length = header[2];
    model->embedding_size = header[3];
    model->block_count = header[4];
    model->head_count = header[5];
    model->is_quantized = header[6];

    /* Merges are consumed by the tokenizer loader below; skip them here. */
    int merge_count;
    if (fread(&merge_count, sizeof(int), 1, file) != 1) { exit(1); }
    fseek(file, (long)merge_count * 3 * sizeof(int), SEEK_CUR);

    int embedding_size = model->embedding_size;
    model->token_embedding = read_floats(file, (long)model->vocabulary_size * embedding_size);
    model->position_embedding = read_floats(file, (long)model->context_length * embedding_size);

    model->blocks = malloc(model->block_count * sizeof(TransformerBlock));
    for (int b = 0; b < model->block_count; b++) {
        TransformerBlock *block = &model->blocks[b];
        block->attention_norm_weight = read_floats(file, embedding_size);
        block->attention_norm_bias = read_floats(file, embedding_size);
        block->qkv_weight = read_matrix(file, 3 * embedding_size, embedding_size, model->is_quantized);
        block->qkv_bias = read_floats(file, 3 * embedding_size);
        block->attention_output_weight = read_matrix(file, embedding_size, embedding_size, model->is_quantized);
        block->attention_output_bias = read_floats(file, embedding_size);
        block->mlp_norm_weight = read_floats(file, embedding_size);
        block->mlp_norm_bias = read_floats(file, embedding_size);
        block->mlp_up_weight = read_matrix(file, 4 * embedding_size, embedding_size, model->is_quantized);
        block->mlp_up_bias = read_floats(file, 4 * embedding_size);
        block->mlp_down_weight = read_matrix(file, embedding_size, 4 * embedding_size, model->is_quantized);
        block->mlp_down_bias = read_floats(file, embedding_size);
    }
    model->final_norm_weight = read_floats(file, embedding_size);
    model->final_norm_bias = read_floats(file, embedding_size);
    fclose(file);
}

/* ------------------------------------------------------- math primitives */

/*
 * Matrix-vector product with bias: output[i] = bias[i] + sum_j W[i][j]*input[j].
 * Dequantizes int8 weights on the fly (multiply by the tensor's scale).
 * This one function is where nearly all the compute goes - Chapter 2, at last
 * running an LLM.
 */
static void matvec(const Matrix *weight, const float *bias, const float *input, float *output) {
    for (int row = 0; row < weight->row_count; row++) {
        float sum = bias ? bias[row] : 0.0f;
        long base = (long)row * weight->column_count;
        if (weight->is_quantized) {
            for (int col = 0; col < weight->column_count; col++) {
                sum += weight->scale * weight->int8_values[base + col] * input[col];
            }
        } else {
            for (int col = 0; col < weight->column_count; col++) {
                sum += weight->float_values[base + col] * input[col];
            }
        }
        output[row] = sum;
    }
}

/* Layer normalization over one vector (Chapter 11), with learned scale+shift. */
static void layer_norm(const float *input, const float *weight, const float *bias,
                       float *output, int size) {
    float mean = 0.0f;
    for (int i = 0; i < size; i++) mean += input[i];
    mean /= size;
    float variance = 0.0f;
    for (int i = 0; i < size; i++) variance += (input[i] - mean) * (input[i] - mean);
    variance /= size;
    float inverse_std = 1.0f / sqrtf(variance + 1e-5f);
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] - mean) * inverse_std * weight[i] + bias[i];
    }
}

/* GELU activation (Chapter 23's transformer convention), tanh approximation. */
static float gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static void softmax(float *values, int count) {
    float maximum = values[0];
    for (int i = 1; i < count; i++) if (values[i] > maximum) maximum = values[i];
    float sum = 0.0f;
    for (int i = 0; i < count; i++) { values[i] = expf(values[i] - maximum); sum += values[i]; }
    for (int i = 0; i < count; i++) values[i] /= sum;
}

/* ------------------------------------------------------- the forward pass */

/*
 * Run the whole model over the token sequence and return the logits for the
 * LAST position (the only ones generation needs).
 *
 * The KV cache from real engines is omitted for clarity: we recompute
 * attention over the whole context each step. For a mini-LLM that is fine and
 * far easier to read; the exercises add the cache.
 *
 * model:        the loaded model
 * token_ids:    the context (length tokens)
 * length:       how many tokens (<= context_length)
 * logits_out:   receives vocabulary_size logits for the final position
 */
static void forward(const Model *model, const int *token_ids, int length, float *logits_out) {
    int embedding_size = model->embedding_size;
    int head_count = model->head_count;
    int head_size = embedding_size / head_count;

    /* Per-position hidden state; scratch buffers reused across blocks. */
    float *hidden = malloc((long)length * embedding_size * sizeof(float));
    float *normed = malloc(embedding_size * sizeof(float));
    float *qkv = malloc(3 * embedding_size * sizeof(float));
    float *queries = malloc((long)length * embedding_size * sizeof(float));
    float *keys = malloc((long)length * embedding_size * sizeof(float));
    float *values = malloc((long)length * embedding_size * sizeof(float));
    float *attention_output = malloc(embedding_size * sizeof(float));
    float *attention_scores = malloc(length * sizeof(float));
    float *projected = malloc(embedding_size * sizeof(float));
    float *mlp_hidden = malloc(4 * embedding_size * sizeof(float));

    /* Embeddings: token meaning + position (Chapter 22/23). */
    for (int t = 0; t < length; t++) {
        for (int i = 0; i < embedding_size; i++) {
            hidden[t * embedding_size + i] =
                model->token_embedding[token_ids[t] * embedding_size + i]
                + model->position_embedding[t * embedding_size + i];
        }
    }

    for (int b = 0; b < model->block_count; b++) {
        const TransformerBlock *block = &model->blocks[b];

        /* --- attention sublayer --- */
        /* First pass: compute Q, K, V for every position (needs the norm). */
        for (int t = 0; t < length; t++) {
            layer_norm(&hidden[t * embedding_size], block->attention_norm_weight,
                       block->attention_norm_bias, normed, embedding_size);
            matvec(&block->qkv_weight, block->qkv_bias, normed, qkv);
            memcpy(&queries[t * embedding_size], qkv, embedding_size * sizeof(float));
            memcpy(&keys[t * embedding_size], qkv + embedding_size, embedding_size * sizeof(float));
            memcpy(&values[t * embedding_size], qkv + 2 * embedding_size, embedding_size * sizeof(float));
        }
        /* Second pass: causal attention per position, per head, then the
         * output projection and residual add (Chapter 22, batched by hand). */
        for (int t = 0; t < length; t++) {
            for (int h = 0; h < head_count; h++) {
                int offset = h * head_size;
                for (int s = 0; s <= t; s++) {   /* causal: only positions <= t */
                    float dot = 0.0f;
                    for (int i = 0; i < head_size; i++) {
                        dot += queries[t * embedding_size + offset + i] * keys[s * embedding_size + offset + i];
                    }
                    attention_scores[s] = dot / sqrtf((float)head_size);
                }
                softmax(attention_scores, t + 1);
                for (int i = 0; i < head_size; i++) {
                    float blended = 0.0f;
                    for (int s = 0; s <= t; s++) {
                        blended += attention_scores[s] * values[s * embedding_size + offset + i];
                    }
                    attention_output[offset + i] = blended;
                }
            }
            matvec(&block->attention_output_weight, block->attention_output_bias, attention_output, projected);
            for (int i = 0; i < embedding_size; i++) {
                hidden[t * embedding_size + i] += projected[i];   /* residual */
            }
        }

        /* --- MLP sublayer (per position, independent) --- */
        for (int t = 0; t < length; t++) {
            layer_norm(&hidden[t * embedding_size], block->mlp_norm_weight,
                       block->mlp_norm_bias, normed, embedding_size);
            matvec(&block->mlp_up_weight, block->mlp_up_bias, normed, mlp_hidden);
            for (int i = 0; i < 4 * embedding_size; i++) mlp_hidden[i] = gelu(mlp_hidden[i]);
            matvec(&block->mlp_down_weight, block->mlp_down_bias, mlp_hidden, projected);
            for (int i = 0; i < embedding_size; i++) {
                hidden[t * embedding_size + i] += projected[i];   /* residual */
            }
        }
    }

    /* Final norm, then the tied output head (token_embedding reused as the
     * projection - Chapter 24's weight tying) for the last position only. */
    layer_norm(&hidden[(length - 1) * embedding_size], model->final_norm_weight,
               model->final_norm_bias, normed, embedding_size);
    for (int v = 0; v < model->vocabulary_size; v++) {
        float sum = 0.0f;
        for (int i = 0; i < embedding_size; i++) {
            sum += model->token_embedding[v * embedding_size + i] * normed[i];
        }
        logits_out[v] = sum;
    }

    free(hidden); free(normed); free(qkv); free(queries); free(keys); free(values);
    free(attention_output); free(attention_scores); free(projected); free(mlp_hidden);
}

/* ------------------------------------------------------- tokenizer (ch. 20) */

typedef struct { int left, right, new_id; } MergeRule;

static MergeRule *merge_rules;
static int merge_count;
static unsigned char token_byte_strings[256 + 4096][64];
static int token_byte_lengths[256 + 4096];

/* Load merges from the same .bin (they sit right after the header). */
static void load_tokenizer(const char *path) {
    FILE *file = fopen(path, "rb");
    fseek(file, 7 * sizeof(int), SEEK_SET);
    if (fread(&merge_count, sizeof(int), 1, file) != 1) { exit(1); }
    merge_rules = malloc(merge_count * sizeof(MergeRule));
    if (fread(merge_rules, sizeof(MergeRule), merge_count, file) != (size_t)merge_count) { exit(1); }
    fclose(file);

    for (int i = 0; i < 256; i++) { token_byte_strings[i][0] = (unsigned char)i; token_byte_lengths[i] = 1; }
    for (int i = 0; i < merge_count; i++) {
        int left = merge_rules[i].left, right = merge_rules[i].right, id = merge_rules[i].new_id;
        token_byte_lengths[id] = token_byte_lengths[left] + token_byte_lengths[right];
        memcpy(token_byte_strings[id], token_byte_strings[left], token_byte_lengths[left]);
        memcpy(token_byte_strings[id] + token_byte_lengths[left], token_byte_strings[right], token_byte_lengths[right]);
    }
}

static int encode_prompt(const char *text, int *token_ids) {
    int count = 0;
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) token_ids[count++] = *p;
    for (int r = 0; r < merge_count; r++) {
        int write = 0;
        for (int read = 0; read < count; read++) {
            if (read + 1 < count && token_ids[read] == merge_rules[r].left && token_ids[read + 1] == merge_rules[r].right) {
                token_ids[write++] = merge_rules[r].new_id; read++;
            } else {
                token_ids[write++] = token_ids[read];
            }
        }
        count = write;
    }
    return count;
}

static void print_token(int token_id) {
    fwrite(token_byte_strings[token_id], 1, token_byte_lengths[token_id], stdout);
    fflush(stdout);
}

/* ------------------------------------------------------- sampling (ch. 23) */

static uint64_t random_state = 1234;
static float random_uniform(void) {
    random_state = random_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)(random_state >> 40) / 16777216.0f;
}

static int sample(float *logits, int count, float temperature, int top_k) {
    for (int i = 0; i < count; i++) logits[i] /= temperature;
    /* top-k: find the k-th largest, mask everything below it. A simple partial
     * selection - fine for our vocabulary size. */
    if (top_k > 0 && top_k < count) {
        float *copy = malloc(count * sizeof(float));
        memcpy(copy, logits, count * sizeof(float));
        for (int k = 0; k < top_k; k++) {
            int best = 0;
            for (int i = 1; i < count; i++) if (copy[i] > copy[best]) best = i;
            copy[best] = -1e30f;
        }
        float threshold = -1e30f;
        for (int i = 0; i < count; i++) if (copy[i] > threshold) threshold = copy[i];
        /* copy now holds the (k+1)-th value at most; anything strictly greater
         * than 'threshold' is in the top k. */
        for (int i = 0; i < count; i++) if (logits[i] <= threshold) logits[i] = -1e30f;
        free(copy);
    }
    softmax(logits, count);
    float threshold = random_uniform(), cumulative = 0.0f;
    for (int i = 0; i < count; i++) { cumulative += logits[i]; if (threshold < cumulative) return i; }
    return count - 1;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.bin> [prompt] [new_tokens]\n", argv[0]);
        return 1;
    }
    const char *model_path = argv[1];
    const char *prompt = argc > 2 ? argv[2] : "It was ";
    int new_tokens = argc > 3 ? atoi(argv[3]) : 200;

    Model model;
    load_model(model_path, &model);
    load_tokenizer(model_path);
    printf("Loaded %s: %s, %d blocks, %d wide, vocab %d, context %d\n\n",
           model_path, model.is_quantized ? "int8" : "float32",
           model.block_count, model.embedding_size, model.vocabulary_size, model.context_length);

    int *token_ids = malloc((model.context_length + 1) * sizeof(int));
    int length = encode_prompt(prompt, token_ids);

    printf("%s", prompt);
    float *logits = malloc(model.vocabulary_size * sizeof(float));
    for (int generated = 0; generated < new_tokens; generated++) {
        int window_start = length > model.context_length ? length - model.context_length : 0;
        forward(&model, token_ids + window_start, length - window_start, logits);
        int next = sample(logits, model.vocabulary_size, 0.8f, 40);
        print_token(next);
        if (length < model.context_length + 1) {
            token_ids[length++] = next;
        } else {
            memmove(token_ids, token_ids + 1, model.context_length * sizeof(int));
            token_ids[model.context_length] = next;
        }
    }
    printf("\n\nGenerated by pure C. The transformer is a file of numbers and these loops.\n");
    return 0;
}
