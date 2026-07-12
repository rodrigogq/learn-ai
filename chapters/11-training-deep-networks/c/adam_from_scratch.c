/*
 * Chapter 11 - the Adam optimizer from scratch, racing plain gradient descent.
 *
 * The task is Chapter 5's apartment regression ON THE RAW FEATURE - the case
 * where plain gradient descent needed 200,000 epochs because the weight and
 * the bias live at wildly different scales. Adam keeps a running average of
 * each parameter's gradient (momentum) and of its squared gradient, then
 * scales every step by that history - so each parameter gets its own
 * effective learning rate. The scale mismatch disappears and the same problem
 * converges about 200x faster.
 *
 * Build and run from the repository root:
 *     make -C chapters/11-training-deep-networks/c
 *     ./chapters/11-training-deep-networks/c/build/adam_from_scratch
 */

#include <math.h>
#include <stdio.h>

#define EXAMPLE_COUNT 12

static const double apartment_sizes_m2[EXAMPLE_COUNT] = {
    30.0, 40.0, 50.0, 55.0, 60.0, 70.0, 75.0, 85.0, 90.0, 100.0, 110.0, 120.0,
};
static const double apartment_prices_k[EXAMPLE_COUNT] = {
    105.0, 144.0, 168.0, 191.0, 192.0, 233.0, 250.0, 271.0, 297.0, 314.0, 352.0, 377.0,
};

/*
 * Mean squared error of the line y = weight*x + bias (Chapter 5's loss).
 */
static double compute_loss(double weight, double bias) {
    double total_squared_error = 0.0;
    for (int example_index = 0; example_index < EXAMPLE_COUNT; example_index++) {
        double prediction_error = weight * apartment_sizes_m2[example_index] + bias
                                - apartment_prices_k[example_index];
        total_squared_error += prediction_error * prediction_error;
    }
    return total_squared_error / EXAMPLE_COUNT;
}

/*
 * Chapter 5's gradients: dL/dw = mean of 2*error*x, dL/db = mean of 2*error.
 *
 * weight, bias:          current parameters
 * gradient_weight_out:   receives dL/dweight
 * gradient_bias_out:     receives dL/dbias
 */
static void compute_gradients(double weight, double bias,
                              double *gradient_weight_out, double *gradient_bias_out) {
    double gradient_weight = 0.0;
    double gradient_bias = 0.0;
    for (int example_index = 0; example_index < EXAMPLE_COUNT; example_index++) {
        double prediction_error = weight * apartment_sizes_m2[example_index] + bias
                                - apartment_prices_k[example_index];
        gradient_weight += prediction_error * apartment_sizes_m2[example_index];
        gradient_bias += prediction_error;
    }
    *gradient_weight_out = 2.0 * gradient_weight / EXAMPLE_COUNT;
    *gradient_bias_out = 2.0 * gradient_bias / EXAMPLE_COUNT;
}

/*
 * Print one progress row if this epoch is in the report list.
 */
static void report_if_scheduled(int epoch, const int *epochs_to_print, int print_count,
                                double weight, double bias) {
    for (int print_index = 0; print_index < print_count; print_index++) {
        if (epochs_to_print[print_index] == epoch) {
            printf("  %7d   %10.2f   %7.3f   %8.3f\n", epoch, compute_loss(weight, bias), weight, bias);
        }
    }
}

/*
 * Plain gradient descent, exactly Chapter 5's loop.
 *
 * learning_rate:     step size (1e-4 is the largest stable value here)
 * number_of_epochs:  how long to run
 */
static void run_plain_gradient_descent(double learning_rate, int number_of_epochs) {
    printf("Plain gradient descent, learning rate %g:\n", learning_rate);
    printf("    epoch         loss         w          b\n");
    const int epochs_to_print[] = {1, 100, 1000, 10000, 100000, 200000};
    double weight = 0.0;
    double bias = 0.0;
    for (int epoch = 1; epoch <= number_of_epochs; epoch++) {
        double gradient_weight, gradient_bias;
        compute_gradients(weight, bias, &gradient_weight, &gradient_bias);
        weight -= learning_rate * gradient_weight;
        bias -= learning_rate * gradient_bias;
        report_if_scheduled(epoch, epochs_to_print, 6, weight, bias);
    }
}

/*
 * Adam from scratch. Per parameter it maintains:
 *   momentum:          a running average of the gradient (direction memory),
 *   squared_gradient:  a running average of the gradient squared (size memory),
 * and steps by  learning_rate * momentum / sqrt(squared_gradient),
 * so a parameter with habitually huge gradients (our weight) gets its steps
 * shrunk, and one with tiny gradients (our bias) gets them boosted.
 *
 * The (1 - decay^t) corrections fix the startup bias: both averages begin at
 * zero and would otherwise underestimate for the first few dozen steps.
 *
 * learning_rate:     base step size (Adam's default scale is ~0.001-1.0
 *                    because steps are normalized; 1.0 suits this tiny problem)
 * number_of_epochs:  how long to run
 */
static void run_adam(double learning_rate, int number_of_epochs) {
    printf("\nAdam, learning rate %g:\n", learning_rate);
    printf("    epoch         loss         w          b\n");
    const int epochs_to_print[] = {1, 10, 100, 1000, 2000};
    const double momentum_decay = 0.9;
    const double squared_gradient_decay = 0.999;
    const double divide_by_zero_guard = 1e-8;

    double weight = 0.0, bias = 0.0;
    double weight_momentum = 0.0, bias_momentum = 0.0;
    double weight_squared_gradient = 0.0, bias_squared_gradient = 0.0;

    for (int epoch = 1; epoch <= number_of_epochs; epoch++) {
        double gradient_weight, gradient_bias;
        compute_gradients(weight, bias, &gradient_weight, &gradient_bias);

        weight_momentum = momentum_decay * weight_momentum + (1.0 - momentum_decay) * gradient_weight;
        bias_momentum = momentum_decay * bias_momentum + (1.0 - momentum_decay) * gradient_bias;
        weight_squared_gradient = squared_gradient_decay * weight_squared_gradient
                                + (1.0 - squared_gradient_decay) * gradient_weight * gradient_weight;
        bias_squared_gradient = squared_gradient_decay * bias_squared_gradient
                              + (1.0 - squared_gradient_decay) * gradient_bias * gradient_bias;

        double momentum_correction = 1.0 - pow(momentum_decay, epoch);
        double squared_correction = 1.0 - pow(squared_gradient_decay, epoch);

        weight -= learning_rate * (weight_momentum / momentum_correction)
                / (sqrt(weight_squared_gradient / squared_correction) + divide_by_zero_guard);
        bias -= learning_rate * (bias_momentum / momentum_correction)
              / (sqrt(bias_squared_gradient / squared_correction) + divide_by_zero_guard);

        report_if_scheduled(epoch, epochs_to_print, 5, weight, bias);
    }
}

int main(void) {
    printf("Chapter 5's raw-feature regression (target: w = 3.0, b = 20). Same problem, two optimizers.\n\n");
    run_plain_gradient_descent(1e-4, 200000);
    run_adam(1.0, 2000);
    printf("\nAdam reached the answer in ~1,000 epochs; plain descent needed ~200,000.\n");
    printf("Per-parameter step normalization is why - the w/b scale mismatch stops mattering.\n");
    return 0;
}
