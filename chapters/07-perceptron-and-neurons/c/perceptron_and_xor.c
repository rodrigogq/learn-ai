/*
 * Chapter 7 - the perceptron, its limit (XOR), and the two-layer fix.
 * Full port of the Python example.
 *
 * Trains a perceptron on AND (converges), OR (converges), XOR (never
 * converges - capped at 20 passes), then runs the hand-wired three-neuron
 * network that computes XOR exactly.
 *
 * Build and run from the repository root:
 *     make -C chapters/07-perceptron-and-neurons/c
 *     ./chapters/07-perceptron-and-neurons/c/build/perceptron_and_xor
 */

#include <stdio.h>

static const int truth_table_inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
static const int and_labels[4] = {0, 0, 0, 1};
static const int or_labels[4] = {0, 1, 1, 1};
static const int xor_labels[4] = {0, 1, 1, 0};

/*
 * History's first activation: 1 if the weighted sum is positive, else 0.
 *
 * weighted_sum: the z value coming out of the weighted sum.
 */
static int step_activation(double weighted_sum) {
    return weighted_sum > 0.0 ? 1 : 0;
}

/*
 * One neuron with the step activation.
 *
 * first_input, second_input:  the two feature values (0 or 1 here)
 * first_weight, second_weight, bias:  the perceptron's three parameters
 */
static int perceptron_predict(int first_input, int second_input,
                              double first_weight, double second_weight, double bias) {
    return step_activation(first_weight * first_input + second_weight * second_input + bias);
}

/*
 * Run Rosenblatt's 1957 learning rule until a full pass makes no mistakes.
 *
 * true_labels:      the four target outputs for the truth-table inputs
 * gate_name:        name printed in the progress table (AND / OR / XOR)
 * maximum_passes:   safety cap - XOR never converges, so we must stop somewhere
 * learning_rate:    how far each wrong prediction nudges the parameters
 *
 * Returns 1 if training converged (a pass with zero mistakes), else 0.
 */
static int train_perceptron(const int *true_labels, const char *gate_name,
                            int maximum_passes, double learning_rate) {
    double first_weight = 0.0;
    double second_weight = 0.0;
    double bias = 0.0;

    printf("Training a perceptron on %s:\n", gate_name);
    printf("  pass   mistakes    w1      w2      b\n");
    for (int pass_number = 1; pass_number <= maximum_passes; pass_number++) {
        int mistakes_this_pass = 0;
        for (int row_index = 0; row_index < 4; row_index++) {
            int first_input = truth_table_inputs[row_index][0];
            int second_input = truth_table_inputs[row_index][1];
            int prediction = perceptron_predict(first_input, second_input,
                                                first_weight, second_weight, bias);
            int prediction_error = true_labels[row_index] - prediction;
            if (prediction_error != 0) {
                mistakes_this_pass++;
                /* The rule in three lines: push each weight toward the correct
                 * answer, but only as much as its input was active. */
                first_weight += learning_rate * prediction_error * first_input;
                second_weight += learning_rate * prediction_error * second_input;
                bias += learning_rate * prediction_error;
            }
        }
        printf("  %4d   %8d  %6.1f  %6.1f  %6.1f\n",
               pass_number, mistakes_this_pass, first_weight, second_weight, bias);
        if (mistakes_this_pass == 0) {
            printf("  -> converged: %s learned in %d passes.\n", gate_name, pass_number);
            return 1;
        }
    }
    printf("  -> did NOT converge in %d passes (and never will - see the chapter).\n", maximum_passes);
    return 0;
}

/*
 * The chapter's hand-wired three-neuron network that computes XOR.
 *
 * first_input, second_input: the two binary inputs.
 *
 * Layer 1 draws two straight lines (an OR gate and an AND gate); the output
 * neuron combines them as "OR but not AND" - which is XOR. No training here:
 * the weights were chosen by staring at the truth table, which is exactly
 * the practice Chapter 8 will make unnecessary.
 */
static int two_layer_network_predict(int first_input, int second_input) {
    int hidden_or_gate = step_activation(first_input + second_input - 0.5);
    int hidden_and_gate = step_activation(first_input + second_input - 1.5);
    return step_activation(hidden_or_gate - hidden_and_gate - 0.5);
}

int main(void) {
    train_perceptron(and_labels, "AND", 20, 0.2);
    printf("\n");
    train_perceptron(or_labels, "OR", 20, 0.2);
    printf("\n");
    train_perceptron(xor_labels, "XOR", 20, 0.2);

    printf("\nThe hand-wired two-layer network on XOR:\n");
    printf("  x1  x2   network   expected\n");
    for (int row_index = 0; row_index < 4; row_index++) {
        int first_input = truth_table_inputs[row_index][0];
        int second_input = truth_table_inputs[row_index][1];
        printf("   %d   %d      %d          %d\n", first_input, second_input,
               two_layer_network_predict(first_input, second_input), xor_labels[row_index]);
    }
    printf("  -> layers solve what one neuron cannot.\n");
    return 0;
}
