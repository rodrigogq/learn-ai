"""Chapter 9 - a full neural network trained on MNIST, from scratch in NumPy.

Architecture: 784 inputs -> 128 hidden neurons (ReLU) -> 10 outputs (softmax).
Training: mini-batch stochastic gradient descent with hand-derived matrix
backpropagation (the chapter explains every formula, and this script verifies
them against a numerical check before training starts).

Run from the repository root:
    .venv/bin/python chapters/09-first-neural-network/python/train_mnist_mlp.py
    .venv/bin/python chapters/09-first-neural-network/python/train_mnist_mlp.py --quick

--quick trains on a 2,000-image subset for one epoch (a few seconds) so you
can verify everything works before the full three-epoch run (a few minutes).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy

# The repo root is two levels up; adding it lets us reuse the shared dataset
# helpers instead of duplicating download code in every chapter.
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from common.data import load_mnist_datasets  # noqa: E402


def load_mnist_as_numpy_arrays():
    """Download MNIST (first run only) and return flat NumPy arrays.

    Returns (training_images, training_labels, test_images, test_labels):
    images are float64 arrays of shape (count, 784) scaled to 0..1, labels are
    integer arrays of shape (count,). torchvision is used purely as a
    downloader here - the actual learning below is NumPy only (PyTorch itself
    enters the course in Chapter 10).
    """
    training_dataset, test_dataset = load_mnist_datasets()

    training_images = training_dataset.data.numpy().reshape(-1, 784).astype(numpy.float64) / 255.0
    training_labels = training_dataset.targets.numpy()
    test_images = test_dataset.data.numpy().reshape(-1, 784).astype(numpy.float64) / 255.0
    test_labels = test_dataset.targets.numpy()
    return training_images, training_labels, test_images, test_labels


def softmax_rows(score_matrix):
    """Turn each row of scores into a probability distribution.

    Arguments:
        score_matrix: array of shape (batch_size, 10), one row of raw scores
            per image.

    Subtracting each row's maximum before exponentiating changes nothing
    mathematically (it cancels in the division) but prevents exp() from
    overflowing on large scores - the standard stability trick.
    """
    shifted_scores = score_matrix - score_matrix.max(axis=1, keepdims=True)
    exponentiated = numpy.exp(shifted_scores)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


class TwoLayerNetwork:
    """The chapter's 784 -> 128 (ReLU) -> 10 (softmax) network."""

    def __init__(self, input_size=784, hidden_size=128, output_size=10, random_seed=42):
        random_generator = numpy.random.default_rng(random_seed)
        # He initialization: weights scaled by sqrt(2 / fan_in) keep the
        # signal's variance stable through a ReLU layer, so deep stacks
        # neither explode nor die at the start (Chapter 11 returns to this).
        self.hidden_weights = random_generator.normal(0.0, numpy.sqrt(2.0 / input_size), (input_size, hidden_size))
        self.hidden_biases = numpy.zeros(hidden_size)
        self.output_weights = random_generator.normal(0.0, numpy.sqrt(2.0 / hidden_size), (hidden_size, output_size))
        self.output_biases = numpy.zeros(output_size)

    def forward(self, image_batch):
        """Run a batch of images through the network.

        Arguments:
            image_batch: array of shape (batch_size, 784).

        Returns (hidden_pre_activation, hidden_activation, class_probabilities);
        the intermediate values are returned because backpropagation needs them.
        """
        hidden_pre_activation = image_batch @ self.hidden_weights + self.hidden_biases
        hidden_activation = numpy.maximum(0.0, hidden_pre_activation)  # ReLU
        output_scores = hidden_activation @ self.output_weights + self.output_biases
        class_probabilities = softmax_rows(output_scores)
        return hidden_pre_activation, hidden_activation, class_probabilities

    def compute_loss_and_gradients(self, image_batch, label_batch):
        """Forward pass, cross-entropy loss, and hand-derived matrix backprop.

        Arguments:
            image_batch: array (batch_size, 784).
            label_batch: integer array (batch_size,) of true digits.

        Returns (loss, gradients) where gradients is a dict keyed like the
        parameter attributes. The formulas are the chapter's; each is
        Chapter 8's local rules applied to a whole layer at once.
        """
        batch_size = image_batch.shape[0]
        hidden_pre_activation, hidden_activation, class_probabilities = self.forward(image_batch)

        probability_of_true_class = class_probabilities[numpy.arange(batch_size), label_batch]
        loss = -numpy.log(numpy.clip(probability_of_true_class, 1e-12, None)).mean()

        # Softmax + cross-entropy cancel to (probabilities - one_hot), the
        # same clean "error" pattern as Chapters 5 and 6.
        one_hot_labels = numpy.zeros_like(class_probabilities)
        one_hot_labels[numpy.arange(batch_size), label_batch] = 1.0
        output_score_gradient = (class_probabilities - one_hot_labels) / batch_size

        gradients = {}
        gradients["output_weights"] = hidden_activation.T @ output_score_gradient
        gradients["output_biases"] = output_score_gradient.sum(axis=0)

        hidden_activation_gradient = output_score_gradient @ self.output_weights.T
        # ReLU's local rule: gradient passes only where the input was positive.
        hidden_pre_activation_gradient = hidden_activation_gradient * (hidden_pre_activation > 0.0)
        gradients["hidden_weights"] = image_batch.T @ hidden_pre_activation_gradient
        gradients["hidden_biases"] = hidden_pre_activation_gradient.sum(axis=0)
        return loss, gradients

    def apply_gradient_step(self, gradients, learning_rate):
        """One gradient-descent update on all four parameter arrays."""
        self.hidden_weights -= learning_rate * gradients["hidden_weights"]
        self.hidden_biases -= learning_rate * gradients["hidden_biases"]
        self.output_weights -= learning_rate * gradients["output_weights"]
        self.output_biases -= learning_rate * gradients["output_biases"]

    def measure_accuracy(self, images, labels):
        """Fraction of images whose highest-probability class is the true digit."""
        _, _, class_probabilities = self.forward(images)
        predicted_digits = class_probabilities.argmax(axis=1)
        return (predicted_digits == labels).mean()


def verify_gradients_numerically(network, image_batch, label_batch):
    """Spot-check the matrix backprop against Chapter 3's central difference.

    Arguments:
        network: the TwoLayerNetwork to check.
        image_batch, label_batch: a small batch to compute the loss on.

    Checking every one of the 101,770 parameters would take minutes, so we
    check a handful of representative ones from each parameter array - enough
    to catch any wrong formula.
    """
    _, analytic_gradients = network.compute_loss_and_gradients(image_batch, label_batch)
    small_step = 1e-6
    print("Gradient spot-check (analytic vs numerical):")
    # Weight (400, 7) touches pixel 400, near the image center, so its
    # gradient is nonzero; border pixels are black in every MNIST image and
    # their weights get exactly zero gradient - try one and see.
    parameters_to_check = [
        ("hidden_weights", network.hidden_weights, (400, 7)),
        ("hidden_biases", network.hidden_biases, (3,)),
        ("output_weights", network.output_weights, (10, 2)),
        ("output_biases", network.output_biases, (9,)),
    ]
    for parameter_name, parameter_array, index in parameters_to_check:
        original_value = parameter_array[index]
        parameter_array[index] = original_value + small_step
        loss_up, _ = network.compute_loss_and_gradients(image_batch, label_batch)
        parameter_array[index] = original_value - small_step
        loss_down, _ = network.compute_loss_and_gradients(image_batch, label_batch)
        parameter_array[index] = original_value
        numerical_gradient = (loss_up - loss_down) / (2 * small_step)
        analytic_gradient = analytic_gradients[parameter_name][index]
        print(f"  {parameter_name}{list(index)}: analytic = {analytic_gradient:+.8f}, numerical = {numerical_gradient:+.8f}")


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--quick", action="store_true",
                                 help="train on a 2,000-image subset for 1 epoch (seconds instead of minutes)")
    parsed_arguments = argument_parser.parse_args()

    print("Loading MNIST (downloads ~12 MB on first run)...")
    training_images, training_labels, test_images, test_labels = load_mnist_as_numpy_arrays()
    number_of_epochs = 5
    if parsed_arguments.quick:
        training_images, training_labels = training_images[:2000], training_labels[:2000]
        test_images, test_labels = test_images[:1000], test_labels[:1000]
        number_of_epochs = 1
        print("(quick mode: 2,000 training images, 1 epoch)")
    print(f"{len(training_images):,} training images, {len(test_images):,} test images")

    network = TwoLayerNetwork()
    print()
    verify_gradients_numerically(network, training_images[:32], training_labels[:32])

    batch_size = 100
    learning_rate = 0.1
    print()
    print(f"Training: batch size {batch_size}, learning rate {learning_rate}")
    print("  epoch   average loss   test accuracy   seconds")
    shuffle_generator = numpy.random.default_rng(123)
    for epoch_number in range(1, number_of_epochs + 1):
        epoch_start_time = time.perf_counter()
        # A fresh shuffle each epoch stops the network from seeing examples
        # in a fixed rhythm it could latch onto.
        shuffled_order = shuffle_generator.permutation(len(training_images))
        loss_sum = 0.0
        batch_count = 0
        for batch_start in range(0, len(training_images) - batch_size + 1, batch_size):
            batch_indices = shuffled_order[batch_start:batch_start + batch_size]
            loss, gradients = network.compute_loss_and_gradients(
                training_images[batch_indices], training_labels[batch_indices]
            )
            network.apply_gradient_step(gradients, learning_rate)
            loss_sum += loss
            batch_count += 1
        epoch_seconds = time.perf_counter() - epoch_start_time
        test_accuracy = network.measure_accuracy(test_images, test_labels)
        print(f"  {epoch_number:>5}   {loss_sum / batch_count:>12.4f}   {test_accuracy:>12.2%}   {epoch_seconds:>7.1f}")

    print()
    final_accuracy = network.measure_accuracy(test_images, test_labels)
    print(f"Final test accuracy: {final_accuracy:.2%} on {len(test_images):,} digits the network never saw.")

    print()
    print("A few individual predictions:")
    _, _, class_probabilities = network.forward(test_images[:5])
    for image_index in range(5):
        predicted_digit = class_probabilities[image_index].argmax()
        confidence = class_probabilities[image_index][predicted_digit]
        print(f"  test image {image_index}: true digit {test_labels[image_index]}, "
              f"predicted {predicted_digit} with confidence {confidence:.1%}")


if __name__ == "__main__":
    main()
