/*
 * Chapter 6 - logistic regression in pure C. Full port of the Python example.
 *
 * Predicts whether a student passes an exam from hours studied: numerical
 * gradient check, gradient-descent training, learned decision boundary, and
 * predictions for three unseen students.
 *
 * Build and run from the repository root:
 *     make -C chapters/06-logistic-regression/c
 *     ./chapters/06-logistic-regression/c/build/train_logistic_regression
 */

#include <math.h>
#include <stdio.h>

#define EXAMPLE_COUNT 12

static const double study_hours[EXAMPLE_COUNT] = {
    0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
};
static const int passed_exam[EXAMPLE_COUNT] = {0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1};

/* Probabilities are clamped this far from exact 0 and 1 before taking logs,
 * because log(0) is minus infinity and would poison the loss. */
static const double probability_clamp = 1e-12;

/*
 * Squash any number into (0, 1): sigma(z) = 1 / (1 + e^(-z)).
 *
 * weighted_sum: the z value, typically weight*feature + bias.
 */
static double sigmoid(double weighted_sum) {
    return 1.0 / (1.0 + exp(-weighted_sum));
}

/*
 * Average surprise of the model at the true labels (Chapter 4's loss).
 *
 * weight, bias:  the model parameters in P(pass) = sigmoid(weight*x + bias)
 */
static double compute_cross_entropy_loss(double weight, double bias) {
    double total_surprise = 0.0;
    for (int example_index = 0; example_index < EXAMPLE_COUNT; example_index++) {
        double predicted_probability = sigmoid(weight * study_hours[example_index] + bias);
        double probability_of_what_happened =
            passed_exam[example_index] ? predicted_probability : 1.0 - predicted_probability;
        if (probability_of_what_happened < probability_clamp) {
            probability_of_what_happened = probability_clamp;
        }
        if (probability_of_what_happened > 1.0 - probability_clamp) {
            probability_of_what_happened = 1.0 - probability_clamp;
        }
        total_surprise += -log(probability_of_what_happened);
    }
    return total_surprise / EXAMPLE_COUNT;
}

/*
 * The chapter's gradients: everything cancels down to (probability - label).
 *
 * weight, bias:          current parameter values
 * gradient_weight_out:   receives the average of (probability - label) * feature
 * gradient_bias_out:     receives the average of (probability - label)
 */
static void compute_loss_gradients(double weight, double bias,
                                   double *gradient_weight_out, double *gradient_bias_out) {
    double gradient_weight = 0.0;
    double gradient_bias = 0.0;
    for (int example_index = 0; example_index < EXAMPLE_COUNT; example_index++) {
        double prediction_error =
            sigmoid(weight * study_hours[example_index] + bias) - passed_exam[example_index];
        gradient_weight += prediction_error * study_hours[example_index];
        gradient_bias += prediction_error;
    }
    *gradient_weight_out = gradient_weight / EXAMPLE_COUNT;
    *gradient_bias_out = gradient_bias / EXAMPLE_COUNT;
}

/*
 * Check the analytic gradients against Chapter 3's central difference.
 *
 * weight, bias:  the point at which both gradient versions are compared
 */
static void verify_gradients_numerically(double weight, double bias) {
    const double small_step = 1e-6;
    double numerical_gradient_weight =
        (compute_cross_entropy_loss(weight + small_step, bias)
       - compute_cross_entropy_loss(weight - small_step, bias)) / (2.0 * small_step);
    double numerical_gradient_bias =
        (compute_cross_entropy_loss(weight, bias + small_step)
       - compute_cross_entropy_loss(weight, bias - small_step)) / (2.0 * small_step);

    double analytic_gradient_weight, analytic_gradient_bias;
    compute_loss_gradients(weight, bias, &analytic_gradient_weight, &analytic_gradient_bias);

    printf("Gradient check at (w=0.5, b=-1):\n");
    printf("  dL/dw: formula = %.6f, numerical = %.6f\n", analytic_gradient_weight, numerical_gradient_weight);
    printf("  dL/db: formula = %.6f, numerical = %.6f\n", analytic_gradient_bias, numerical_gradient_bias);
}

int main(void) {
    verify_gradients_numerically(0.5, -1.0);

    printf("\nTraining (learning rate 0.5):\n");
    printf("  epoch    loss        w         b   boundary(h)\n");
    double weight = 0.0;
    double bias = 0.0;
    const double learning_rate = 0.5;
    const int epochs_to_print[] = {0, 10, 100, 1000, 5000};
    for (int epoch_number = 0; epoch_number <= 5000; epoch_number++) {
        for (int print_index = 0; print_index < 5; print_index++) {
            if (epochs_to_print[print_index] == epoch_number) {
                if (weight != 0.0) {
                    printf("  %5d  %.4f  %7.3f  %8.3f  %8.2f\n", epoch_number,
                           compute_cross_entropy_loss(weight, bias), weight, bias, -bias / weight);
                } else {
                    printf("  %5d  %.4f  %7.3f  %8.3f         -\n", epoch_number,
                           compute_cross_entropy_loss(weight, bias), weight, bias);
                }
            }
        }
        double gradient_weight, gradient_bias;
        compute_loss_gradients(weight, bias, &gradient_weight, &gradient_bias);
        weight = weight - learning_rate * gradient_weight;
        bias = bias - learning_rate * gradient_bias;
    }

    printf("\nLearned model: P(pass) = sigmoid(%.3f * hours %c %.3f)\n",
           weight, bias >= 0.0 ? '+' : '-', fabs(bias));
    printf("Decision boundary: %.2f hours of study\n", -bias / weight);

    printf("\nInference on three new students:\n");
    const double new_student_hours[] = {2.0, 3.7, 5.0};
    for (int student_index = 0; student_index < 3; student_index++) {
        double pass_probability = sigmoid(weight * new_student_hours[student_index] + bias);
        printf("  %.1f h -> P(pass) = %.3f -> predict %s\n", new_student_hours[student_index],
               pass_probability, pass_probability >= 0.5 ? "pass" : "fail");
    }
    return 0;
}
