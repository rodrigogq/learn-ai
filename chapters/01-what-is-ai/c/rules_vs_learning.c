/*
 * Chapter 1 - the same problem solved twice: hand-written rules vs learning.
 *
 * Full C port of python/rules_vs_learning.py: classify fruit (apple/orange)
 * from weight and surface smoothness. Version 1 uses guessed rules; version 2
 * learns each class's average fruit (nearest-centroid classifier) from
 * labeled examples. Both are judged on fruits neither saw during training.
 *
 * Build and run from the repository root:
 *     make -C chapters/01-what-is-ai/c
 *     ./chapters/01-what-is-ai/c/build/rules_vs_learning
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

#define LABEL_APPLE 0
#define LABEL_ORANGE 1

typedef struct {
    double weight_in_grams;
    double surface_smoothness;  /* 0 = very rough, 1 = very smooth */
    int true_label;             /* LABEL_APPLE or LABEL_ORANGE */
} Fruit;

/* Same tiny datasets as the Python version, so the outputs match. */
static const Fruit training_fruits[] = {
    {120.0, 0.80, LABEL_APPLE},  {130.0, 0.90, LABEL_APPLE},
    {140.0, 0.85, LABEL_APPLE},  {130.0, 0.85, LABEL_APPLE},
    {160.0, 0.40, LABEL_ORANGE}, {170.0, 0.45, LABEL_ORANGE},
    {180.0, 0.40, LABEL_ORANGE}, {190.0, 0.43, LABEL_ORANGE},
};

static const Fruit test_fruits[] = {
    {118.0, 0.90, LABEL_APPLE},  {160.0, 0.80, LABEL_APPLE},
    {135.0, 0.88, LABEL_APPLE},  {155.0, 0.45, LABEL_ORANGE},
    {185.0, 0.38, LABEL_ORANGE}, {165.0, 0.50, LABEL_ORANGE},
};

#define TRAINING_FRUIT_COUNT ((int)(sizeof(training_fruits) / sizeof(training_fruits[0])))
#define TEST_FRUIT_COUNT ((int)(sizeof(test_fruits) / sizeof(test_fruits[0])))

static const char *label_name(int label) {
    return label == LABEL_APPLE ? "apple" : "orange";
}

/*
 * Classify a fruit using rules a programmer guessed by eye.
 *
 * weight_in_grams:     the fruit's weight
 * surface_smoothness:  0 (very rough) to 1 (very smooth)
 *
 * Returns LABEL_APPLE or LABEL_ORANGE.
 */
static int classify_with_hand_written_rules(double weight_in_grams, double surface_smoothness) {
    /* The threshold 150 is a guess and the rule ignores smoothness entirely -
     * nobody checked it against data, which is exactly what this chapter
     * demonstrates. The unused parameter keeps both classifiers callable the
     * same way. */
    (void)surface_smoothness;
    if (weight_in_grams > 150.0) {
        return LABEL_ORANGE;
    }
    return LABEL_APPLE;
}

/*
 * THE LEARNING STEP: compute the average fruit (centroid) of one class.
 *
 * fruits:              array of labeled training fruits
 * fruit_count:         number of elements in fruits
 * label_to_average:    which class to average (LABEL_APPLE or LABEL_ORANGE)
 * average_weight_out:      receives the class's average weight
 * average_smoothness_out:  receives the class's average smoothness
 *
 * The two numbers written to the output pointers are the model's PARAMETERS:
 * they come from data, not from a person.
 */
static void learn_class_average(const Fruit *fruits, int fruit_count, int label_to_average,
                                double *average_weight_out, double *average_smoothness_out) {
    double weight_sum = 0.0;
    double smoothness_sum = 0.0;
    int matching_fruit_count = 0;
    for (int fruit_index = 0; fruit_index < fruit_count; fruit_index++) {
        if (fruits[fruit_index].true_label == label_to_average) {
            weight_sum += fruits[fruit_index].weight_in_grams;
            smoothness_sum += fruits[fruit_index].surface_smoothness;
            matching_fruit_count++;
        }
    }
    *average_weight_out = weight_sum / matching_fruit_count;
    *average_smoothness_out = smoothness_sum / matching_fruit_count;
}

/*
 * Distance from a fruit to a class average, with weight rescaled so both
 * features count. Smoothness lives in [0, 1] while weight lives near 150;
 * without the /100 the distance would be decided by weight alone.
 *
 * weight_in_grams, surface_smoothness:  the fruit being classified
 * average_weight, average_smoothness:   one class's learned centroid
 */
static double distance_to_class_average(double weight_in_grams, double surface_smoothness,
                                        double average_weight, double average_smoothness) {
    double weight_difference = (weight_in_grams - average_weight) / 100.0;
    double smoothness_difference = surface_smoothness - average_smoothness;
    return sqrt(weight_difference * weight_difference + smoothness_difference * smoothness_difference);
}

int main(void) {
    printf("Training data: %d fruits with known labels.\n\n", TRAINING_FRUIT_COUNT);

    double apple_average_weight, apple_average_smoothness;
    double orange_average_weight, orange_average_smoothness;
    learn_class_average(training_fruits, TRAINING_FRUIT_COUNT, LABEL_APPLE,
                        &apple_average_weight, &apple_average_smoothness);
    learn_class_average(training_fruits, TRAINING_FRUIT_COUNT, LABEL_ORANGE,
                        &orange_average_weight, &orange_average_smoothness);

    printf("LEARNING STEP (computing the parameters from data):\n");
    printf("  average apple  -> weight %.1f g, smoothness %.2f\n", apple_average_weight, apple_average_smoothness);
    printf("  average orange -> weight %.1f g, smoothness %.2f\n\n", orange_average_weight, orange_average_smoothness);

    printf("Judging %d new fruits both programs have never seen:\n\n", TEST_FRUIT_COUNT);
    printf("  fruit (g, smooth)   true      rules say   learned model says\n");

    int rule_based_correct_count = 0;
    int learned_model_correct_count = 0;
    for (int fruit_index = 0; fruit_index < TEST_FRUIT_COUNT; fruit_index++) {
        const Fruit *fruit = &test_fruits[fruit_index];

        int rule_based_prediction = classify_with_hand_written_rules(fruit->weight_in_grams,
                                                                     fruit->surface_smoothness);

        double distance_to_apple = distance_to_class_average(fruit->weight_in_grams, fruit->surface_smoothness,
                                                             apple_average_weight, apple_average_smoothness);
        double distance_to_orange = distance_to_class_average(fruit->weight_in_grams, fruit->surface_smoothness,
                                                              orange_average_weight, orange_average_smoothness);
        int learned_model_prediction = distance_to_apple < distance_to_orange ? LABEL_APPLE : LABEL_ORANGE;

        rule_based_correct_count += rule_based_prediction == fruit->true_label;
        learned_model_correct_count += learned_model_prediction == fruit->true_label;

        printf("  (%3.0f, %.2f)         %-9s %-11s %s\n",
               fruit->weight_in_grams, fruit->surface_smoothness,
               label_name(fruit->true_label), label_name(rule_based_prediction),
               label_name(learned_model_prediction));
    }

    printf("\n");
    printf("Hand-written rules:  %d/%d correct\n", rule_based_correct_count, TEST_FRUIT_COUNT);
    printf("Learned model:       %d/%d correct\n", learned_model_correct_count, TEST_FRUIT_COUNT);
    return 0;
}
