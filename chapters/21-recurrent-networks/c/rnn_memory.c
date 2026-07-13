/*
 * Chapter 21 - the RNN recurrence in pure C, and a hand-wired demonstration
 * that the hidden state really is memory.
 *
 * Part 1 runs the vanilla RNN equation new_state = tanh(W*x + U*state + b)
 * over a sequence, printing the state at every step so you can watch
 * information persist.
 *
 * Part 2 hand-wires a TWO-NEURON recurrent network that solves a task no
 * feedforward network could: judging whether a bracket string like "(()())"
 * is balanced. Neuron 1 counts depth (its self-connection is the counter);
 * neuron 2 latches an error flag if depth ever goes negative. No training -
 * the point is seeing exactly WHAT recurrence buys: state across time.
 *
 * Build and run from the repository root:
 *     make -C chapters/21-recurrent-networks/c
 *     ./chapters/21-recurrent-networks/c/build/rnn_memory
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

/*
 * One step of a vanilla RNN cell.
 *
 * input_vector:    input_size numbers for this time step
 * previous_state:  hidden_size numbers carried from the last step
 * input_weights:   hidden_size x input_size, row-major
 * state_weights:   hidden_size x hidden_size, row-major (the memory wiring)
 * biases:          hidden_size numbers
 * new_state_out:   receives hidden_size numbers
 * input_size, hidden_size: dimensions
 */
static void rnn_cell_step(const double *input_vector, const double *previous_state,
                          const double *input_weights, const double *state_weights,
                          const double *biases, double *new_state_out,
                          int input_size, int hidden_size) {
    for (int neuron = 0; neuron < hidden_size; neuron++) {
        double weighted_sum = biases[neuron];
        for (int i = 0; i < input_size; i++) {
            weighted_sum += input_weights[neuron * input_size + i] * input_vector[i];
        }
        for (int i = 0; i < hidden_size; i++) {
            weighted_sum += state_weights[neuron * hidden_size + i] * previous_state[i];
        }
        new_state_out[neuron] = tanh(weighted_sum);
    }
}

/*
 * Part 2's bracket checker. Inputs are one-hot: x = (is_open, is_close).
 * The recurrence here is LINEAR (no tanh) so the counting is exact - a
 * deliberate simplification that keeps every number readable. Trained RNNs
 * approximate this same mechanism with saturating tanh neurons.
 *
 * bracket_string: e.g. "(()())"
 *
 * Prints the two-number state after each character and the verdict.
 */
static void check_brackets(const char *bracket_string) {
    double depth_counter = 0.0;    /* neuron 1: += 1 for '(' , -= 1 for ')' */
    double error_latch = 0.0;      /* neuron 2: jumps to 1 if depth < 0, then self-sustains */

    printf("  \"%s\"\n", bracket_string);
    printf("    step  char  depth  error-latch\n");
    for (int position = 0; bracket_string[position] != '\0'; position++) {
        double is_open = bracket_string[position] == '(' ? 1.0 : 0.0;
        double is_close = bracket_string[position] == ')' ? 1.0 : 0.0;

        depth_counter = depth_counter + is_open - is_close;
        /* The latch: once on, its self-connection keeps it on - one bad
         * moment is remembered forever. This is recurrent memory at its
         * starkest. */
        if (depth_counter < 0.0 || error_latch > 0.5) {
            error_latch = 1.0;
        }
        printf("    %4d   %c    %5.0f  %11.0f\n",
               position, bracket_string[position], depth_counter, error_latch);
    }
    int balanced = error_latch < 0.5 && fabs(depth_counter) < 0.5;
    printf("    verdict: %s\n\n", balanced ? "BALANCED" : "not balanced");
}

int main(void) {
    printf("1. The recurrence, running: state = tanh(W*x + U*state + b)\n");
    printf("   One hidden neuron, input weight 2, self-weight 2.5, bias 0.\n");
    printf("   We poke it once at step 2 (input 1) and never again:\n\n");

    double input_weights[1] = {2.0};
    double state_weights[1] = {2.5};
    double biases[1] = {0.0};
    double state[1] = {0.0};
    double new_state[1];

    printf("    step  input   state afterwards\n");
    for (int step = 0; step < 8; step++) {
        double input_vector[1] = {step == 2 ? 1.0 : 0.0};
        rnn_cell_step(input_vector, state, input_weights, state_weights, biases,
                      new_state, 1, 1);
        state[0] = new_state[0];
        printf("    %4d   %.0f      %+.4f\n", step, input_vector[0], state[0]);
    }
    printf("\n   The poke at step 2 echoes forever: the self-connection (weight 2.5,\n");
    printf("   saturating tanh) holds the state near +1. That persistence is memory -\n");
    printf("   and with self-weight 0.5 instead, the echo would fade in a few steps\n");
    printf("   (try it): the vanishing-memory problem in miniature.\n\n");

    printf("2. A hand-wired two-neuron recurrent bracket checker\n\n");
    check_brackets("(()())");
    check_brackets("(()))(");
    check_brackets("((()");

    printf("No feedforward network of ANY size can do this for arbitrary lengths -\n");
    printf("it has no state to count with. Recurrence is what buys memory.\n");
    return 0;
}
