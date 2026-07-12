"""Chapter 7 - the perceptron, its limit (XOR), and the two-layer fix.

Four demonstrations:
  1. train a perceptron on AND  - converges in a few passes,
  2. train a perceptron on OR   - converges,
  3. train a perceptron on XOR  - the mistake count cycles forever (capped at 20 passes),
  4. run the hand-wired two-layer network that computes XOR exactly.

Run from the repository root:
    .venv/bin/python chapters/07-perceptron-and-neurons/python/perceptron_and_xor.py
"""

TRUTH_TABLE_INPUTS = [(0, 0), (0, 1), (1, 0), (1, 1)]
AND_LABELS = [0, 0, 0, 1]
OR_LABELS = [0, 1, 1, 1]
XOR_LABELS = [0, 1, 1, 0]


def step_activation(weighted_sum):
    """History's first activation: 1 if the weighted sum is positive, else 0.

    Arguments:
        weighted_sum: the z value coming out of the weighted sum.
    """
    return 1 if weighted_sum > 0 else 0


def perceptron_predict(first_input, second_input, weights_and_bias):
    """One neuron with the step activation.

    Arguments:
        first_input, second_input: the two feature values (0 or 1 here).
        weights_and_bias: list [w1, w2, b], the perceptron's three parameters.
    """
    first_weight, second_weight, bias = weights_and_bias
    return step_activation(first_weight * first_input + second_weight * second_input + bias)


def train_perceptron(true_labels, gate_name, maximum_passes=20, learning_rate=0.2):
    """Run Rosenblatt's 1957 learning rule until a full pass makes no mistakes.

    Arguments:
        true_labels: the four target outputs for the truth-table inputs.
        gate_name: name printed in the progress table (AND / OR / XOR).
        maximum_passes: safety cap - XOR never converges, so we must stop somewhere.
        learning_rate: how far each wrong prediction nudges the parameters.

    Returns True if training converged (a pass with zero mistakes), else False.
    """
    weights_and_bias = [0.0, 0.0, 0.0]
    print(f"Training a perceptron on {gate_name}:")
    print("  pass   mistakes    w1      w2      b")
    for pass_number in range(1, maximum_passes + 1):
        mistakes_this_pass = 0
        for (first_input, second_input), true_label in zip(TRUTH_TABLE_INPUTS, true_labels):
            prediction = perceptron_predict(first_input, second_input, weights_and_bias)
            prediction_error = true_label - prediction
            if prediction_error != 0:
                mistakes_this_pass += 1
                # The rule in three lines: push each weight toward the correct
                # answer, but only as much as its input was active.
                weights_and_bias[0] += learning_rate * prediction_error * first_input
                weights_and_bias[1] += learning_rate * prediction_error * second_input
                weights_and_bias[2] += learning_rate * prediction_error
        print(f"  {pass_number:>4}   {mistakes_this_pass:>8}  {weights_and_bias[0]:>6.1f}  {weights_and_bias[1]:>6.1f}  {weights_and_bias[2]:>6.1f}")
        if mistakes_this_pass == 0:
            print(f"  -> converged: {gate_name} learned in {pass_number} passes.")
            return True
    print(f"  -> did NOT converge in {maximum_passes} passes (and never will - see the chapter).")
    return False


def two_layer_network_predict(first_input, second_input):
    """The chapter's hand-wired three-neuron network that computes XOR.

    Arguments:
        first_input, second_input: the two binary inputs.

    Layer 1 draws two straight lines (an OR gate and an AND gate); the output
    neuron combines them as "OR but not AND" - which is XOR. No training here:
    the weights were chosen by staring at the truth table, which is exactly
    the practice Chapter 8 will make unnecessary.
    """
    hidden_or_gate = step_activation(first_input + second_input - 0.5)
    hidden_and_gate = step_activation(first_input + second_input - 1.5)
    return step_activation(hidden_or_gate - hidden_and_gate - 0.5)


def main():
    train_perceptron(AND_LABELS, "AND")
    print()
    train_perceptron(OR_LABELS, "OR")
    print()
    train_perceptron(XOR_LABELS, "XOR")

    print()
    print("The hand-wired two-layer network on XOR:")
    print("  x1  x2   network   expected")
    for (first_input, second_input), expected_output in zip(TRUTH_TABLE_INPUTS, XOR_LABELS):
        network_output = two_layer_network_predict(first_input, second_input)
        print(f"   {first_input}   {second_input}      {network_output}          {expected_output}")
    print("  -> layers solve what one neuron cannot.")


if __name__ == "__main__":
    main()
