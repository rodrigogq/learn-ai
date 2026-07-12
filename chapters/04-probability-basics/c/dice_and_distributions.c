/*
 * Chapter 4 - probability verified by brute force, in pure C.
 *
 * Full port of the Python example: two-dice distribution by simulation vs
 * exact counting, expected values by sampling, and cross-entropy grading of
 * two weather forecasters.
 *
 * Note the fixed seed (srand(42)): every reader sees the same "random"
 * numbers, so the printed output is reproducible. Seeding is a habit kept
 * through the whole course - it is how you debug anything involving
 * randomness.
 *
 * Build and run from the repository root:
 *     make -C chapters/04-probability-basics/c
 *     ./chapters/04-probability-basics/c/build/dice_and_distributions
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Return one fair die roll, an integer from 1 to 6.
 *
 * rand() returns 0..RAND_MAX; the modulo folds it into 0..5 and the +1 shifts
 * to 1..6. The tiny bias this introduces (RAND_MAX is not a multiple of 6) is
 * far below what a million rolls can detect, so it is fine for this chapter.
 */
static int roll_one_die(void) {
    return rand() % 6 + 1;
}

/*
 * Return the exact probability of a given two-dice sum by counting pairs.
 *
 * dice_sum: the sum in question, 2 through 12.
 *
 * Out of 36 equally likely pairs, the number summing to s is 6 - |7 - s|.
 */
static double exact_two_dice_probability(int dice_sum) {
    int number_of_pairs = 6 - abs(7 - dice_sum);
    return number_of_pairs / 36.0;
}

/*
 * Compute a forecaster's cross-entropy: their average surprise at reality.
 *
 * rain_probabilities_given:  probability the forecaster gave to "rain", per day
 * it_actually_rained:        1 where it rained, 0 where it did not, per day
 * number_of_days:            how many days both arrays cover
 *
 * Returns the average of -log(probability given to what actually happened).
 * On a dry day the probability given to the actual outcome is 1 - p_rain,
 * which is why both branches appear below.
 */
static double compute_cross_entropy_for_rain_forecasts(const double *rain_probabilities_given,
                                                       const int *it_actually_rained,
                                                       int number_of_days) {
    double total_surprise = 0.0;
    for (int day_index = 0; day_index < number_of_days; day_index++) {
        double probability_given_to_actual_outcome =
            it_actually_rained[day_index] ? rain_probabilities_given[day_index]
                                          : 1.0 - rain_probabilities_given[day_index];
        total_surprise += -log(probability_given_to_actual_outcome);
    }
    return total_surprise / number_of_days;
}

int main(void) {
    srand(42);

    const int number_of_rolls = 1000000;
    printf("1. Two-dice distribution: %d simulated rolls vs exact counting\n", number_of_rolls);

    long sum_counts[13] = {0};
    for (int roll_index = 0; roll_index < number_of_rolls; roll_index++) {
        sum_counts[roll_one_die() + roll_one_die()]++;
    }
    printf("  sum   simulated   exact\n");
    for (int dice_sum = 2; dice_sum <= 12; dice_sum++) {
        printf("  %3d   %.5f     %.5f\n", dice_sum,
               (double)sum_counts[dice_sum] / number_of_rolls,
               exact_two_dice_probability(dice_sum));
    }

    printf("\n2. Expected values estimated by sampling (exact: 3.5 and 7.0)\n");
    const int sample_sizes[] = {100, 10000, 1000000};
    for (int size_index = 0; size_index < 3; size_index++) {
        int number_of_samples = sample_sizes[size_index];
        long single_die_total = 0;
        long two_dice_total = 0;
        for (int sample_index = 0; sample_index < number_of_samples; sample_index++) {
            single_die_total += roll_one_die();
            two_dice_total += roll_one_die() + roll_one_die();
        }
        printf("  %9d samples: one die = %.4f, two dice = %.4f\n", number_of_samples,
               (double)single_die_total / number_of_samples,
               (double)two_dice_total / number_of_samples);
    }

    printf("\n3. Cross-entropy: grading two weather forecasters (it rained on days 1, 2, 5)\n");
    const int it_actually_rained[] = {1, 1, 0, 0, 1};
    const double forecaster_a_probabilities[] = {0.8, 0.9, 0.2, 0.1, 0.6};
    const double forecaster_b_probabilities[] = {0.5, 0.5, 0.5, 0.5, 0.5};
    printf("  forecaster A (confident and mostly right): %.2f\n",
           compute_cross_entropy_for_rain_forecasts(forecaster_a_probabilities, it_actually_rained, 5));
    printf("  forecaster B (always says 50/50):          %.2f\n",
           compute_cross_entropy_for_rain_forecasts(forecaster_b_probabilities, it_actually_rained, 5));
    printf("  Lower is better: cross-entropy rewards confident correctness.\n");
    return 0;
}
