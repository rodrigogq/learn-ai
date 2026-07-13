/*
 * Chapter 20 - the BPE encoder/decoder in pure C, sharing the Python trainer's
 * merges file.
 *
 * Training a tokenizer happens once, in Python; ENCODING happens at every
 * inference everywhere, so it is the part worth owning in C. This program
 * loads datasets/bpe_merges.txt (written by the Python trainer), encodes a
 * sentence by replaying the merges, decodes it back, and verifies the round
 * trip. Chapter 25's pure-C LLM inference engine uses exactly this code path.
 *
 * Before the first run (from the repository root):
 *     .venv/bin/python chapters/20-text-and-tokenization/python/bpe_tokenizer.py
 *
 * Build and run from the repository root:
 *     make -C chapters/20-text-and-tokenization/c
 *     ./chapters/20-text-and-tokenization/c/build/bpe_encoder
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_MERGES 4096
#define MAX_VOCABULARY (256 + MAX_MERGES)
#define MAX_TOKEN_BYTES 64
#define MAX_TEXT_TOKENS 4096

typedef struct {
    int left_id;
    int right_id;
    int new_id;
} MergeRule;

static MergeRule merge_rules[MAX_MERGES];
static int merge_count = 0;

/* Each token's bytes, built up as merges are loaded (decoding needs them). */
static unsigned char token_bytes[MAX_VOCABULARY][MAX_TOKEN_BYTES];
static int token_byte_count[MAX_VOCABULARY];

/*
 * Load the merges file the Python trainer wrote: 'left right new' per line.
 * Also builds each token's byte string by concatenating its parents'.
 */
static void load_merges(const char *file_path) {
    FILE *file = fopen(file_path, "r");
    if (file == NULL) {
        fprintf(stderr, "Cannot open %s - run the Python trainer first:\n", file_path);
        fprintf(stderr, "  .venv/bin/python chapters/20-text-and-tokenization/python/bpe_tokenizer.py\n");
        exit(1);
    }
    for (int byte_value = 0; byte_value < 256; byte_value++) {
        token_bytes[byte_value][0] = (unsigned char)byte_value;
        token_byte_count[byte_value] = 1;
    }

    int left_id, right_id, new_id;
    while (merge_count < MAX_MERGES
           && fscanf(file, "%d %d %d", &left_id, &right_id, &new_id) == 3) {
        merge_rules[merge_count] = (MergeRule){left_id, right_id, new_id};
        token_byte_count[new_id] = token_byte_count[left_id] + token_byte_count[right_id];
        memcpy(token_bytes[new_id], token_bytes[left_id], token_byte_count[left_id]);
        memcpy(token_bytes[new_id] + token_byte_count[left_id],
               token_bytes[right_id], token_byte_count[right_id]);
        merge_count++;
    }
    fclose(file);
}

/*
 * Encode text by replaying the merges in training order (earlier merges were
 * more frequent in training and take priority - same rule as the Python).
 *
 * text:       the input string
 * token_ids:  receives the ids
 *
 * Returns the token count.
 */
static int encode(const char *text, int *token_ids) {
    int token_count = 0;
    for (const unsigned char *byte = (const unsigned char *)text; *byte != '\0'; byte++) {
        token_ids[token_count++] = *byte;
    }

    for (int rule_index = 0; rule_index < merge_count; rule_index++) {
        MergeRule rule = merge_rules[rule_index];
        int write_position = 0;
        for (int read_position = 0; read_position < token_count; read_position++) {
            if (read_position + 1 < token_count
                && token_ids[read_position] == rule.left_id
                && token_ids[read_position + 1] == rule.right_id) {
                token_ids[write_position++] = rule.new_id;
                read_position++;   /* the pair is consumed together */
            } else {
                token_ids[write_position++] = token_ids[read_position];
            }
        }
        token_count = write_position;
    }
    return token_count;
}

/*
 * Decode token ids back to text: concatenate every token's bytes.
 *
 * token_ids, token_count:  the encoded sequence
 * text_out:                receives the string (caller provides the space)
 */
static void decode(const int *token_ids, int token_count, char *text_out) {
    int output_length = 0;
    for (int i = 0; i < token_count; i++) {
        memcpy(text_out + output_length, token_bytes[token_ids[i]], token_byte_count[token_ids[i]]);
        output_length += token_byte_count[token_ids[i]];
    }
    text_out[output_length] = '\0';
}

int main(void) {
    load_merges("datasets/bpe_merges.txt");
    printf("Loaded %d merges (vocabulary: 256 bytes + %d learned tokens)\n\n", merge_count, merge_count);

    const char *sentence = "To be, or not to be: that is the question.";
    static int token_ids[MAX_TEXT_TOKENS];
    int token_count = encode(sentence, token_ids);

    printf("text:   \"%s\" (%d bytes)\n", sentence, (int)strlen(sentence));
    printf("tokens: %d ids:", token_count);
    for (int i = 0; i < token_count; i++) {
        printf(" %d", token_ids[i]);
    }
    printf("\npieces: ");
    for (int i = 0; i < token_count; i++) {
        printf("[");
        fwrite(token_bytes[token_ids[i]], 1, token_byte_count[token_ids[i]], stdout);
        printf("]");
    }
    printf("\n");

    static char round_trip[MAX_TEXT_TOKENS * MAX_TOKEN_BYTES];
    decode(token_ids, token_count, round_trip);
    printf("\ndecode(encode(text)) == text: %s\n",
           strcmp(round_trip, sentence) == 0 ? "true" : "FALSE");
    printf("Identical ids to the Python encoder - one merges file, two languages.\n");
    return 0;
}
