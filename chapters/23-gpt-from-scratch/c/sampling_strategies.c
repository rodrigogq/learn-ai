/*
 * Chapter 23 - sampling strategies in pure C: the decision a language model
 * makes at every single token.
 *
 * A GPT's forward pass ends in logits - raw scores over the vocabulary. HOW
 * you pick the next token from them changes everything about the output.
 * This program takes one fixed, realistic logit vector and applies the four
 * standard strategies, sampling many times so the differences are visible
 * in the counts:
 *
 *   greedy       - always the argmax: deterministic, quickly repetitive
 *   temperature  - divide logits by T before softmax: T<1 safe, T>1 wild
 *   top-k        - keep only the k best, renormalize, sample
 *   (and T=1 plain sampling as the baseline)
 *
 * Chapter 25's inference engine uses exactly these functions.
 *
 * Build and run from the repository root:
 *     make -C chapters/23-gpt-from-scratch/c
 *     ./chapters/23-gpt-from-scratch/c/build/sampling_strategies
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define VOCABULARY_SIZE 8

/* A realistic next-token situation: two strong candidates, a few plausible
 * ones, and a long tail. Imagine the context "To be, or not to ..." */
static const char *candidate_tokens[VOCABULARY_SIZE] = {
    "be", "die", "sleep", "live", "me", "the", "xylophone", "%",
};
static const double logits[VOCABULARY_SIZE] = {5.0, 3.5, 3.0, 2.5, 1.5, 1.0, -2.0, -3.0};

/* Chapter 9's deterministic generator. */
static double pseudo_random_uniform(uint64_t *state) {
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(*state >> 11) * (1.0 / 9007199254740992.0);
}

/*
 * Softmax with temperature (Chapter 9's softmax; the divisor is the knob).
 *
 * input_logits:   raw scores
 * temperature:    divisor; 1 = the model's honest distribution
 * probabilities:  receives VOCABULARY_SIZE values summing to 1
 */
static void softmax_with_temperature(const double *input_logits, double temperature,
                                     double *probabilities) {
    double maximum = -1e300;
    for (int i = 0; i < VOCABULARY_SIZE; i++) {
        if (input_logits[i] / temperature > maximum) {
            maximum = input_logits[i] / temperature;
        }
    }
    double sum = 0.0;
    for (int i = 0; i < VOCABULARY_SIZE; i++) {
        probabilities[i] = exp(input_logits[i] / temperature - maximum);
        sum += probabilities[i];
    }
    for (int i = 0; i < VOCABULARY_SIZE; i++) {
        probabilities[i] /= sum;
    }
}

/*
 * Sample one index from a probability distribution: draw a uniform number,
 * walk the cumulative sum until it is exceeded.
 */
static int sample_from_distribution(const double *probabilities, uint64_t *random_state) {
    double threshold = pseudo_random_uniform(random_state);
    double cumulative = 0.0;
    for (int i = 0; i < VOCABULARY_SIZE; i++) {
        cumulative += probabilities[i];
        if (threshold < cumulative) {
            return i;
        }
    }
    return VOCABULARY_SIZE - 1;
}

/*
 * Top-k filtering: zero every probability outside the k largest, renormalize.
 * Kills the long tail (where the embarrassing tokens live) while keeping
 * genuine variety among the plausible candidates.
 */
static void apply_top_k(double *probabilities, int k) {
    /* Implemented as "zero the (VOCABULARY_SIZE - k) smallest", which is the
     * same thing as keeping the k largest. */
    for (int removals = 0; removals < VOCABULARY_SIZE - k; removals++) {
        int smallest = -1;
        for (int i = 0; i < VOCABULARY_SIZE; i++) {
            if (probabilities[i] > 0.0 && (smallest == -1 || probabilities[i] < probabilities[smallest])) {
                smallest = i;
            }
        }
        probabilities[smallest] = 0.0;
    }
    double sum = 0.0;
    for (int i = 0; i < VOCABULARY_SIZE; i++) {
        sum += probabilities[i];
    }
    for (int i = 0; i < VOCABULARY_SIZE; i++) {
        probabilities[i] /= sum;
    }
}

/*
 * Run one strategy 10,000 times and print how often each token came out.
 */
static void report_strategy(const char *strategy_name, double temperature, int top_k) {
    double probabilities[VOCABULARY_SIZE];
    softmax_with_temperature(logits, temperature, probabilities);
    if (top_k > 0) {
        apply_top_k(probabilities, top_k);
    }

    int counts[VOCABULARY_SIZE] = {0};
    uint64_t random_state = 42;
    for (int trial = 0; trial < 10000; trial++) {
        counts[sample_from_distribution(probabilities, &random_state)]++;
    }

    printf("  %-22s", strategy_name);
    for (int i = 0; i < VOCABULARY_SIZE; i++) {
        printf(" %5.1f%%", counts[i] / 100.0);
    }
    printf("\n");
}

int main(void) {
    printf("The situation: logits for the next token after \"To be, or not to ...\"\n\n");
    printf("  %-22s", "token:");
    for (int i = 0; i < VOCABULARY_SIZE; i++) {
        printf(" %6s", candidate_tokens[i]);
    }
    printf("\n  %-22s", "logit:");
    for (int i = 0; i < VOCABULARY_SIZE; i++) {
        printf(" %6.1f", logits[i]);
    }
    printf("\n\nWhat 10,000 samples of each strategy actually produce:\n\n");

    report_strategy("greedy (T -> 0)", 0.01, 0);
    report_strategy("temperature 0.5", 0.5, 0);
    report_strategy("temperature 1.0", 1.0, 0);
    report_strategy("temperature 1.5", 1.5, 0);
    report_strategy("top-k 3 (at T=1)", 1.0, 3);

    printf("\nRead the last column ('%%', the garbage token): plain T=1.0 sampling picks it\n");
    printf("occasionally, T=1.5 picks it often, and top-k NEVER does - which is why real\n");
    printf("systems combine a moderate temperature with top-k (or its cousin, nucleus/top-p).\n");
    return 0;
}
