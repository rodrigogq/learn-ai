/*
 * Chapter 0 - first C program of the course.
 *
 * Verifies that the C toolchain works and computes one weighted sum by hand.
 * As the chapter explains, the weighted sum (values times weights, added up,
 * plus an offset) is the single operation modern AI is built from - here you
 * meet it as plain arithmetic, nothing more.
 *
 * Build and run from the repository root:
 *     make -C chapters/00-setup/c
 *     ./chapters/00-setup/c/build/hello_ai
 */

#include <stdio.h>

/*
 * Combine values into one number, weighting each value by its importance.
 *
 * values:            array of value_count numbers to combine
 * weights:           array of value_count importance factors, one per value
 * offset:            fixed amount added at the end, independent of the values
 * value_count:       how many elements values and weights hold
 *
 * Returns values[0]*weights[0] + values[1]*weights[1] + ... + offset.
 */
double compute_weighted_sum(const double *values, const double *weights,
                            double offset, int value_count) {
    double accumulated_sum = 0.0;
    for (int value_index = 0; value_index < value_count; value_index++) {
        accumulated_sum += values[value_index] * weights[value_index];
    }
    return accumulated_sum + offset;
}

int main(void) {
    printf("Hello from C (standard C11)!\n");
    printf("\n");
    printf("This machine uses %zu bytes for a 'double' (the number type used everywhere in this course).\n",
           sizeof(double));

    double example_values[] = {0.5, 0.3};
    double example_weights[] = {0.8, -0.2};
    double example_offset = 0.1;
    double weighted_sum_result = compute_weighted_sum(example_values, example_weights,
                                                      example_offset, 2);

    printf("\n");
    printf("A weighted sum, computed by hand:\n");
    printf("  values:  [%.2f, %.2f]\n", example_values[0], example_values[1]);
    printf("  weights: [%.2f, %.2f]\n", example_weights[0], example_weights[1]);
    printf("  offset:  %.2f\n", example_offset);
    printf("  result = 0.5*0.8 + 0.3*(-0.2) + 0.1 = %.3f\n", weighted_sum_result);

    printf("\n");
    printf("Your C toolchain is ready for the course.\n");
    return 0;
}
