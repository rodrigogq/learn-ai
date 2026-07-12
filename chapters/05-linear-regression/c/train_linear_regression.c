/*
 * Chapter 5 - linear regression trained from scratch, in pure C.
 *
 * Full port of the Python example: gradient check, training on the raw
 * feature (200,000 epochs), training on the standardized feature (300
 * epochs), and pricing an unseen 80 m^2 apartment.
 *
 * Build and run from the repository root:
 *     make -C chapters/05-linear-regression/c
 *     ./chapters/05-linear-regression/c/build/train_linear_regression
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
 * Average of squared prediction errors for the line y_hat = weight*x + bias.
 *
 * weight, bias:   the line's two parameters
 * input_values:   array of EXAMPLE_COUNT feature values (apartment sizes)
 * true_values:    array of EXAMPLE_COUNT true answers (prices)
 */
static double compute_mean_squared_error(double weight, double bias,
                                         const double *input_values, const double *true_values) {
    double total_squared_error = 0.0;
    for (int example_index = 0; example_index < EXAMPLE_COUNT; example_index++) {
        double prediction_error = weight * input_values[example_index] + bias - true_values[example_index];
        total_squared_error += prediction_error * prediction_error;
    }
    return total_squared_error / EXAMPLE_COUNT;
}

/*
 * The hand-derived MSE gradients from the chapter text:
 *   dL/dweight = (2/n) * sum of (error * input)
 *   dL/dbias   = (2/n) * sum of (error)
 *
 * weight, bias:          current parameter values
 * input_values:          array of EXAMPLE_COUNT feature values
 * true_values:           array of EXAMPLE_COUNT true answers
 * gradient_weight_out:   receives dL/dweight
 * gradient_bias_out:     receives dL/dbias
 */
static void compute_loss_gradients(double weight, double bias,
                                   const double *input_values, const double *true_values,
                                   double *gradient_weight_out, double *gradient_bias_out) {
    double gradient_weight = 0.0;
    double gradient_bias = 0.0;
    for (int example_index = 0; example_index < EXAMPLE_COUNT; example_index++) {
        double prediction_error = weight * input_values[example_index] + bias - true_values[example_index];
        gradient_weight += prediction_error * input_values[example_index];
        gradient_bias += prediction_error;
    }
    *gradient_weight_out = 2.0 * gradient_weight / EXAMPLE_COUNT;
    *gradient_bias_out = 2.0 * gradient_bias / EXAMPLE_COUNT;
}

/*
 * Check the analytic gradients against Chapter 3's central difference.
 * Trust nothing derived by hand until a numerical check agrees with it.
 *
 * weight, bias:  the point at which both gradient versions are compared
 */
static void verify_gradients_numerically(double weight, double bias) {
    const double small_step = 1e-6;
    double numerical_gradient_weight =
        (compute_mean_squared_error(weight + small_step, bias, apartment_sizes_m2, apartment_prices_k)
       - compute_mean_squared_error(weight - small_step, bias, apartment_sizes_m2, apartment_prices_k))
        / (2.0 * small_step);
    double numerical_gradient_bias =
        (compute_mean_squared_error(weight, bias + small_step, apartment_sizes_m2, apartment_prices_k)
       - compute_mean_squared_error(weight, bias - small_step, apartment_sizes_m2, apartment_prices_k))
        / (2.0 * small_step);

    double analytic_gradient_weight, analytic_gradient_bias;
    compute_loss_gradients(weight, bias, apartment_sizes_m2, apartment_prices_k,
                           &analytic_gradient_weight, &analytic_gradient_bias);

    printf("Gradient check at (w=1, b=5):\n");
    printf("  dL/dw: formula = %.4f, numerical = %.4f\n", analytic_gradient_weight, numerical_gradient_weight);
    printf("  dL/db: formula = %.4f, numerical = %.4f\n", analytic_gradient_bias, numerical_gradient_bias);
}

/*
 * Run the four-step training loop (forward, loss, gradients, update).
 *
 * input_values:      feature values to train on (raw or standardized)
 * true_values:       true answers
 * learning_rate:     step size for both parameter updates
 * number_of_epochs:  how many full passes over the data to run
 * epochs_to_print:   array of epoch numbers to show in the progress table
 * print_count:       how many entries epochs_to_print holds
 * weight_out, bias_out:  receive the trained parameters
 */
static void train_with_gradient_descent(const double *input_values, const double *true_values,
                                        double learning_rate, int number_of_epochs,
                                        const int *epochs_to_print, int print_count,
                                        double *weight_out, double *bias_out) {
    double weight = 0.0;
    double bias = 0.0;
    printf("  epoch      loss         w         b\n");
    for (int epoch_number = 0; epoch_number <= number_of_epochs; epoch_number++) {
        for (int print_index = 0; print_index < print_count; print_index++) {
            if (epochs_to_print[print_index] == epoch_number) {
                printf("  %7d  %10.1f  %8.3f  %8.3f\n", epoch_number,
                       compute_mean_squared_error(weight, bias, input_values, true_values),
                       weight, bias);
            }
        }
        double gradient_weight, gradient_bias;
        compute_loss_gradients(weight, bias, input_values, true_values,
                               &gradient_weight, &gradient_bias);
        weight = weight - learning_rate * gradient_weight;
        bias = bias - learning_rate * gradient_bias;
    }
    *weight_out = weight;
    *bias_out = bias;
}

/*
 * Shift and scale the apartment sizes to mean 0 and standard deviation 1.
 *
 * standardized_out:            receives EXAMPLE_COUNT standardized values
 * mean_out, standard_deviation_out:  receive the statistics, needed later to
 *                                    convert the learned line back to raw units
 */
static void standardize_sizes(double *standardized_out, double *mean_out, double *standard_deviation_out) {
    double sum_of_values = 0.0;
    for (int example_index = 0; example_index < EXAMPLE_COUNT; example_index++) {
        sum_of_values += apartment_sizes_m2[example_index];
    }
    double mean_value = sum_of_values / EXAMPLE_COUNT;

    double sum_of_squared_deviations = 0.0;
    for (int example_index = 0; example_index < EXAMPLE_COUNT; example_index++) {
        double deviation = apartment_sizes_m2[example_index] - mean_value;
        sum_of_squared_deviations += deviation * deviation;
    }
    double standard_deviation = sqrt(sum_of_squared_deviations / EXAMPLE_COUNT);

    for (int example_index = 0; example_index < EXAMPLE_COUNT; example_index++) {
        standardized_out[example_index] = (apartment_sizes_m2[example_index] - mean_value) / standard_deviation;
    }
    *mean_out = mean_value;
    *standard_deviation_out = standard_deviation;
}

int main(void) {
    verify_gradients_numerically(1.0, 5.0);

    printf("\nTraining on RAW sizes (learning rate 1e-4 - watch the bias crawl):\n");
    int raw_epochs_to_print[] = {0, 1, 10, 10000, 100000, 200000};
    double weight_raw, bias_raw;
    train_with_gradient_descent(apartment_sizes_m2, apartment_prices_k,
                                1e-4, 200000, raw_epochs_to_print, 6, &weight_raw, &bias_raw);

    printf("\nTraining on STANDARDIZED sizes (learning rate 0.1 - same loop, 300 epochs):\n");
    double standardized_sizes[EXAMPLE_COUNT];
    double size_mean, size_standard_deviation;
    standardize_sizes(standardized_sizes, &size_mean, &size_standard_deviation);
    int standardized_epochs_to_print[] = {0, 1, 10, 100, 300};
    double weight_standardized, bias_standardized;
    train_with_gradient_descent(standardized_sizes, apartment_prices_k,
                                0.1, 300, standardized_epochs_to_print, 5,
                                &weight_standardized, &bias_standardized);

    /* The model learned prices from z = (size - mean) / std. Substituting z
     * back gives price = (w/std)*size + (b - w*mean/std): the raw-unit line. */
    double weight_raw_units = weight_standardized / size_standard_deviation;
    double bias_raw_units = bias_standardized - weight_standardized * size_mean / size_standard_deviation;
    printf("\nLine recovered in raw units: price = %.3f * size + %.3f\n", weight_raw_units, bias_raw_units);

    double predicted_price = weight_raw_units * 80.0 + bias_raw_units;
    printf("Inference: an 80 m^2 apartment should cost about $%.0f,000\n", predicted_price);
    return 0;
}
