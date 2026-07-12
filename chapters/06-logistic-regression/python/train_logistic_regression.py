"""Chapter 6 - logistic regression: the first trained classifier.

Predicts whether a student passes an exam from hours studied:
  1. verify the cross-entropy gradients against a numerical check,
  2. train with gradient descent (the same loop as Chapter 5),
  3. report the learned decision boundary,
  4. predict pass probabilities for three unseen students.

Run from the repository root:
    .venv/bin/python chapters/06-logistic-regression/python/train_logistic_regression.py
"""

import math

STUDY_HOURS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
PASSED_EXAM = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1]

# Probabilities are clamped this far away from exact 0 and 1 before taking
# logs, because log(0) is minus infinity and would poison the loss. Every real
# ML framework applies the same safety rail internally.
PROBABILITY_CLAMP = 1e-12


def sigmoid(weighted_sum):
    """Squash any number into (0, 1): sigma(z) = 1 / (1 + e^(-z)).

    Arguments:
        weighted_sum: the z value, typically weight*feature + bias.
    """
    return 1.0 / (1.0 + math.exp(-weighted_sum))


def compute_cross_entropy_loss(weight, bias, feature_values, true_labels):
    """Average surprise of the model at the true labels (Chapter 4's loss).

    Arguments:
        weight, bias: the model parameters in P(pass) = sigmoid(weight*x + bias).
        feature_values: list of study hours.
        true_labels: list of 0/1 outcomes, same length.
    """
    total_surprise = 0.0
    for feature_value, true_label in zip(feature_values, true_labels):
        predicted_probability = sigmoid(weight * feature_value + bias)
        probability_of_what_happened = predicted_probability if true_label == 1 else 1.0 - predicted_probability
        clamped_probability = min(max(probability_of_what_happened, PROBABILITY_CLAMP), 1.0 - PROBABILITY_CLAMP)
        total_surprise += -math.log(clamped_probability)
    return total_surprise / len(feature_values)


def compute_loss_gradients(weight, bias, feature_values, true_labels):
    """The chapter's gradients: everything cancels down to (probability - label).

    Arguments:
        weight, bias: current parameter values.
        feature_values: list of study hours.
        true_labels: list of 0/1 outcomes.

    Returns (dL/dweight, dL/dbias):
        dL/dweight = average of (predicted_probability - label) * feature
        dL/dbias   = average of (predicted_probability - label)
    """
    number_of_examples = len(feature_values)
    gradient_weight = 0.0
    gradient_bias = 0.0
    for feature_value, true_label in zip(feature_values, true_labels):
        prediction_error = sigmoid(weight * feature_value + bias) - true_label
        gradient_weight += prediction_error * feature_value
        gradient_bias += prediction_error
    return gradient_weight / number_of_examples, gradient_bias / number_of_examples


def verify_gradients_numerically(weight, bias, small_step=1e-6):
    """Check the analytic gradients against Chapter 3's central difference.

    Arguments:
        weight, bias: the point at which both gradient versions are compared.
        small_step: the h of the central difference.
    """
    numerical_gradient_weight = (
        compute_cross_entropy_loss(weight + small_step, bias, STUDY_HOURS, PASSED_EXAM)
        - compute_cross_entropy_loss(weight - small_step, bias, STUDY_HOURS, PASSED_EXAM)
    ) / (2 * small_step)
    numerical_gradient_bias = (
        compute_cross_entropy_loss(weight, bias + small_step, STUDY_HOURS, PASSED_EXAM)
        - compute_cross_entropy_loss(weight, bias - small_step, STUDY_HOURS, PASSED_EXAM)
    ) / (2 * small_step)
    analytic_gradient_weight, analytic_gradient_bias = compute_loss_gradients(
        weight, bias, STUDY_HOURS, PASSED_EXAM
    )
    print("Gradient check at (w=0.5, b=-1):")
    print(f"  dL/dw: formula = {analytic_gradient_weight:.6f}, numerical = {numerical_gradient_weight:.6f}")
    print(f"  dL/db: formula = {analytic_gradient_bias:.6f}, numerical = {numerical_gradient_bias:.6f}")


def main():
    verify_gradients_numerically(0.5, -1.0)

    print()
    print("Training (learning rate 0.5):")
    print("  epoch    loss        w         b   boundary(h)")
    weight = 0.0
    bias = 0.0
    learning_rate = 0.5
    epochs_to_print = {0, 10, 100, 1000, 5000}
    for epoch_number in range(5001):
        if epoch_number in epochs_to_print:
            current_loss = compute_cross_entropy_loss(weight, bias, STUDY_HOURS, PASSED_EXAM)
            boundary_text = f"{-bias / weight:>8.2f}" if weight != 0.0 else "       -"
            print(f"  {epoch_number:>5}  {current_loss:.4f}  {weight:>7.3f}  {bias:>8.3f}  {boundary_text}")
        gradient_weight, gradient_bias = compute_loss_gradients(weight, bias, STUDY_HOURS, PASSED_EXAM)
        weight = weight - learning_rate * gradient_weight
        bias = bias - learning_rate * gradient_bias

    print()
    decision_boundary_hours = -bias / weight
    bias_sign = "+" if bias >= 0 else "-"
    print(f"Learned model: P(pass) = sigmoid({weight:.3f} * hours {bias_sign} {abs(bias):.3f})")
    print(f"Decision boundary: {decision_boundary_hours:.2f} hours of study")

    print()
    print("Inference on three new students:")
    for hours_studied in (2.0, 3.7, 5.0):
        pass_probability = sigmoid(weight * hours_studied + bias)
        decision = "pass" if pass_probability >= 0.5 else "fail"
        print(f"  {hours_studied:.1f} h -> P(pass) = {pass_probability:.3f} -> predict {decision}")


if __name__ == "__main__":
    main()
