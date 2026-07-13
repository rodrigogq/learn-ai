/*
 * Chapter 22 - one attention head in pure C, reproducing the chapter's
 * worked example exactly.
 *
 * Attention is four small steps: scores = Q K^T / sqrt(d), optional causal
 * mask, softmax per row, output = weights V. Everything here is Chapter 2
 * matrix code plus one softmax - which is the chapter's real message: the
 * mechanism behind every modern LLM fits on one screen of C.
 *
 * Build and run from the repository root:
 *     make -C chapters/22-attention-and-transformers/c
 *     ./chapters/22-attention-and-transformers/c/build/attention_head
 */

#include <math.h>
#include <stdio.h>

#define TOKEN_COUNT 3
#define KEY_SIZE 2
#define VALUE_SIZE 2

/*
 * One attention head, forward pass.
 *
 * queries: TOKEN_COUNT x KEY_SIZE   - what each position is looking for
 * keys:    TOKEN_COUNT x KEY_SIZE   - what each position offers to be found by
 * values:  TOKEN_COUNT x VALUE_SIZE - what each position hands over if chosen
 * use_causal_mask: hide the future (position i attends only to j <= i)
 * output:  receives TOKEN_COUNT x VALUE_SIZE
 * attention_weights: receives TOKEN_COUNT x TOKEN_COUNT (for inspection)
 */
static void attention_head(const double queries[TOKEN_COUNT][KEY_SIZE],
                           const double keys[TOKEN_COUNT][KEY_SIZE],
                           const double values[TOKEN_COUNT][VALUE_SIZE],
                           int use_causal_mask,
                           double output[TOKEN_COUNT][VALUE_SIZE],
                           double attention_weights[TOKEN_COUNT][TOKEN_COUNT]) {
    for (int query_position = 0; query_position < TOKEN_COUNT; query_position++) {
        /* Scores: this query dotted with every key, scaled by sqrt(d) so the
         * numbers stay in softmax's comfortable range as dimensions grow. */
        double scores[TOKEN_COUNT];
        for (int key_position = 0; key_position < TOKEN_COUNT; key_position++) {
            double dot_product = 0.0;
            for (int dimension = 0; dimension < KEY_SIZE; dimension++) {
                dot_product += queries[query_position][dimension] * keys[key_position][dimension];
            }
            scores[key_position] = dot_product / sqrt((double)KEY_SIZE);
            if (use_causal_mask && key_position > query_position) {
                scores[key_position] = -1e30;   /* minus infinity, effectively: softmax -> 0 */
            }
        }

        /* Softmax over the scores (Chapter 9's, with the max-subtraction guard). */
        double maximum_score = scores[0];
        for (int i = 1; i < TOKEN_COUNT; i++) {
            if (scores[i] > maximum_score) maximum_score = scores[i];
        }
        double exponent_sum = 0.0;
        for (int i = 0; i < TOKEN_COUNT; i++) {
            attention_weights[query_position][i] = exp(scores[i] - maximum_score);
            exponent_sum += attention_weights[query_position][i];
        }
        for (int i = 0; i < TOKEN_COUNT; i++) {
            attention_weights[query_position][i] /= exponent_sum;
        }

        /* Output: the weighted average of every position's value vector. */
        for (int dimension = 0; dimension < VALUE_SIZE; dimension++) {
            double weighted_sum = 0.0;
            for (int key_position = 0; key_position < TOKEN_COUNT; key_position++) {
                weighted_sum += attention_weights[query_position][key_position]
                              * values[key_position][dimension];
            }
            output[query_position][dimension] = weighted_sum;
        }
    }
}

static void print_matrix(const char *label, int rows, int columns, const double *matrix) {
    printf("%s\n", label);
    for (int row = 0; row < rows; row++) {
        printf("   ");
        for (int column = 0; column < columns; column++) {
            printf(" %7.3f", matrix[row * columns + column]);
        }
        printf("\n");
    }
}

int main(void) {
    /* The chapter's worked example - same numbers as the Python. */
    double queries[TOKEN_COUNT][KEY_SIZE] = {{1, 0}, {0, 1}, {1, 1}};
    double keys[TOKEN_COUNT][KEY_SIZE] = {{1, 0}, {0, 1}, {1, 0}};
    double values[TOKEN_COUNT][VALUE_SIZE] = {{1, 0}, {10, 0}, {0, 10}};

    double output[TOKEN_COUNT][VALUE_SIZE];
    double attention_weights[TOKEN_COUNT][TOKEN_COUNT];

    printf("One attention head on the chapter's worked example (no mask):\n\n");
    attention_head(queries, keys, values, 0, output, attention_weights);
    print_matrix("attention weights (rows sum to 1):", TOKEN_COUNT, TOKEN_COUNT, &attention_weights[0][0]);
    print_matrix("output (weights @ V):", TOKEN_COUNT, VALUE_SIZE, &output[0][0]);

    printf("\nWith the causal mask (a language model may not peek at the future):\n\n");
    attention_head(queries, keys, values, 1, output, attention_weights);
    print_matrix("attention weights:", TOKEN_COUNT, TOKEN_COUNT, &attention_weights[0][0]);
    print_matrix("output:", TOKEN_COUNT, VALUE_SIZE, &output[0][0]);

    printf("\nSame numbers as the Python - and nothing here but dot products and a softmax.\n");
    return 0;
}
