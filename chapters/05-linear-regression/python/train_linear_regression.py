"""Chapter 5 - linear regression trained from scratch with gradient descent.

The full supervised-learning recipe on 12 apartments (size -> price):
  1. verify the hand-derived MSE gradients against a numerical check,
  2. train on the raw feature (slow bias convergence, 200,000 epochs),
  3. train on the standardized feature (converges in ~300 epochs),
  4. use the trained model to price an unseen 80 m^2 apartment.

Run from the repository root:
    .venv/bin/python chapters/05-linear-regression/python/train_linear_regression.py
"""

APARTMENT_SIZES_M2 = [30.0, 40.0, 50.0, 55.0, 60.0, 70.0, 75.0, 85.0, 90.0, 100.0, 110.0, 120.0]
APARTMENT_PRICES_K = [105.0, 144.0, 168.0, 191.0, 192.0, 233.0, 250.0, 271.0, 297.0, 314.0, 352.0, 377.0]


def compute_mean_squared_error(weight, bias, input_values, true_values):
    """Average of squared prediction errors for the line y_hat = weight*x + bias.

    Arguments:
        weight, bias: the line's two parameters.
        input_values: list of feature values (apartment sizes).
        true_values: list of true answers (prices), same length.
    """
    total_squared_error = 0.0
    for input_value, true_value in zip(input_values, true_values):
        prediction_error = weight * input_value + bias - true_value
        total_squared_error += prediction_error ** 2
    return total_squared_error / len(input_values)


def compute_loss_gradients(weight, bias, input_values, true_values):
    """The hand-derived MSE gradients from the chapter text.

    Arguments:
        weight, bias: current parameter values.
        input_values: list of feature values.
        true_values: list of true answers.

    Returns (dL/dweight, dL/dbias):
        dL/dweight = (2/n) * sum of (error * input)   - errors on big inputs push harder
        dL/dbias   = (2/n) * sum of (error)           - the average error, doubled
    """
    number_of_examples = len(input_values)
    gradient_weight = 0.0
    gradient_bias = 0.0
    for input_value, true_value in zip(input_values, true_values):
        prediction_error = weight * input_value + bias - true_value
        gradient_weight += prediction_error * input_value
        gradient_bias += prediction_error
    return 2.0 * gradient_weight / number_of_examples, 2.0 * gradient_bias / number_of_examples


def verify_gradients_numerically(weight, bias, input_values, true_values, small_step=1e-6):
    """Check the analytic gradients against Chapter 3's central difference.

    Arguments:
        weight, bias: the point at which both gradient versions are compared.
        input_values, true_values: the dataset.
        small_step: the h of the central difference.

    Trust nothing derived by hand until a numerical check agrees with it -
    this habit becomes essential when the models grow (Chapter 8).
    """
    numerical_gradient_weight = (
        compute_mean_squared_error(weight + small_step, bias, input_values, true_values)
        - compute_mean_squared_error(weight - small_step, bias, input_values, true_values)
    ) / (2 * small_step)
    numerical_gradient_bias = (
        compute_mean_squared_error(weight, bias + small_step, input_values, true_values)
        - compute_mean_squared_error(weight, bias - small_step, input_values, true_values)
    ) / (2 * small_step)
    analytic_gradient_weight, analytic_gradient_bias = compute_loss_gradients(
        weight, bias, input_values, true_values
    )
    print("Gradient check at (w=1, b=5):")
    print(f"  dL/dw: formula = {analytic_gradient_weight:.4f}, numerical = {numerical_gradient_weight:.4f}")
    print(f"  dL/db: formula = {analytic_gradient_bias:.4f}, numerical = {numerical_gradient_bias:.4f}")


def train_with_gradient_descent(input_values, true_values, learning_rate, number_of_epochs, epochs_to_print):
    """Run the four-step training loop and return the final (weight, bias).

    Arguments:
        input_values: feature values to train on (raw or standardized).
        true_values: true answers.
        learning_rate: step size for both parameter updates.
        number_of_epochs: how many full passes over the data to run.
        epochs_to_print: which epochs to show in the progress table.
    """
    weight = 0.0
    bias = 0.0
    print("  epoch      loss         w         b")
    for epoch_number in range(number_of_epochs + 1):
        if epoch_number in epochs_to_print:
            current_loss = compute_mean_squared_error(weight, bias, input_values, true_values)
            print(f"  {epoch_number:>7}  {current_loss:>10.1f}  {weight:>8.3f}  {bias:>8.3f}")
        gradient_weight, gradient_bias = compute_loss_gradients(weight, bias, input_values, true_values)
        weight = weight - learning_rate * gradient_weight
        bias = bias - learning_rate * gradient_bias
    return weight, bias


def standardize_values(values):
    """Shift and scale a list to mean 0 and standard deviation 1.

    Arguments:
        values: the raw feature values.

    Returns (standardized_values, mean, standard_deviation). The mean and
    standard deviation are returned because converting the learned line back
    to raw units (and standardizing future inputs) needs them.
    """
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    standard_deviation = variance ** 0.5
    standardized = [(value - mean_value) / standard_deviation for value in values]
    return standardized, mean_value, standard_deviation


def main():
    verify_gradients_numerically(1.0, 5.0, APARTMENT_SIZES_M2, APARTMENT_PRICES_K)

    print()
    print("Training on RAW sizes (learning rate 1e-4 - watch the bias crawl):")
    weight_raw, bias_raw = train_with_gradient_descent(
        APARTMENT_SIZES_M2, APARTMENT_PRICES_K,
        learning_rate=1e-4, number_of_epochs=200_000,
        epochs_to_print={0, 1, 10, 10_000, 100_000, 200_000},
    )
    print(f"  -> converged: price = {weight_raw:.3f} * size + {bias_raw:.3f}")
    print("     (The loss stops at ~24.4, not 0: the 12 points do not lie on a")
    print("      perfectly straight line, so 24.4 IS the best any line can do.)")

    print()
    print("Training on STANDARDIZED sizes (learning rate 0.1 - same loop, 300 epochs):")
    standardized_sizes, size_mean, size_standard_deviation = standardize_values(APARTMENT_SIZES_M2)
    weight_standardized, bias_standardized = train_with_gradient_descent(
        standardized_sizes, APARTMENT_PRICES_K,
        learning_rate=0.1, number_of_epochs=300,
        epochs_to_print={0, 1, 10, 100, 300},
    )
    print(f"  -> converged in standardized units to w={weight_standardized:.3f}, b={bias_standardized:.3f}.")
    print("     These look nothing like the raw run's 3 and 20 because the feature")
    print("     was shifted and scaled - but they are the SAME line, as we now show.")

    # The model learned prices from z = (size - mean) / std. Substituting z
    # back gives price = (w/std)*size + (b - w*mean/std): the raw-unit line.
    weight_raw_units = weight_standardized / size_standard_deviation
    bias_raw_units = bias_standardized - weight_standardized * size_mean / size_standard_deviation
    print()
    print(f"Standardized line converted back to raw units: price = {weight_raw_units:.3f} * size + {bias_raw_units:.3f}")
    print(f"Raw run found:                                 price = {weight_raw:.3f} * size + {bias_raw:.3f}")
    print("Both runs found the same line - one just took 600x more epochs to get there.")

    predicted_price = weight_raw_units * 80.0 + bias_raw_units
    print(f"\nInference: an 80 m^2 apartment should cost about ${predicted_price:.0f},000")


if __name__ == "__main__":
    main()
