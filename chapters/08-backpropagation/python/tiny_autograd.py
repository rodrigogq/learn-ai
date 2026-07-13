"""Chapter 8 - a tiny automatic differentiation engine, then XOR learned with it.

Four demonstrations:
  1. the chain rule worked example, verified numerically,
  2. backpropagation through the graph L = (a*b + c)^2 - the figure's numbers,
  3. the engine's gradients re-verified with Chapter 3's numerical checker,
  4. a 2-3-1 tanh network trained on XOR: the weights Chapter 7 hand-wired
     are now LEARNED.

Run from the repository root:
    .venv/bin/python chapters/08-backpropagation/python/tiny_autograd.py
"""

import math


class TrackedValue:
    """A number that remembers how it was computed, so gradients can flow back.

    Each TrackedValue stores four things:
      data      - the actual number (the forward-pass value),
      gradient  - dLoss/d(this value), filled in by the backward pass,
      the parent values it was computed from, and
      a small function holding the LOCAL derivative rule of the operation
      that created it (set by __add__, __mul__, tanh below).

    Doing arithmetic on TrackedValues builds the computation graph as a side
    effect; calling .run_backward_pass() on the final value then applies the
    chain rule through the whole graph.
    """

    def __init__(self, data, parent_values=()):
        self.data = data
        self.gradient = 0.0
        self._parent_values = parent_values
        self._propagate_gradient_to_parents = lambda: None

    @staticmethod
    def _wrap_if_plain_number(candidate):
        """Allow mixing plain floats into expressions: they become leaf nodes."""
        return candidate if isinstance(candidate, TrackedValue) else TrackedValue(float(candidate))

    def __add__(self, other_value):
        other_value = TrackedValue._wrap_if_plain_number(other_value)
        result = TrackedValue(self.data + other_value.data, (self, other_value))

        def propagate_addition_gradient():
            # Addition's local derivative is 1 toward both parents, so the
            # incoming gradient passes through unchanged. "+=" (not "=")
            # because a value used in several places collects gradient from
            # every path.
            self.gradient += result.gradient
            other_value.gradient += result.gradient

        result._propagate_gradient_to_parents = propagate_addition_gradient
        return result

    def __mul__(self, other_value):
        other_value = TrackedValue._wrap_if_plain_number(other_value)
        result = TrackedValue(self.data * other_value.data, (self, other_value))

        def propagate_multiplication_gradient():
            # Multiplication's local derivative toward each parent is the
            # OTHER parent's value: d(a*b)/da = b and d(a*b)/db = a.
            self.gradient += other_value.data * result.gradient
            other_value.gradient += self.data * result.gradient

        result._propagate_gradient_to_parents = propagate_multiplication_gradient
        return result

    def __sub__(self, other_value):
        # Built from existing operations (a - b = a + b*(-1)), so subtraction
        # needs no backward rule of its own - the graph handles it.
        return self + TrackedValue._wrap_if_plain_number(other_value) * -1.0

    def tanh(self):
        tanh_of_data = math.tanh(self.data)
        result = TrackedValue(tanh_of_data, (self,))

        def propagate_tanh_gradient():
            # tanh's slope is expressible from its own OUTPUT: 1 - tanh(z)^2.
            self.gradient += (1.0 - tanh_of_data * tanh_of_data) * result.gradient

        result._propagate_gradient_to_parents = propagate_tanh_gradient
        return result

    def run_backward_pass(self):
        """Backpropagation: seed dL/dL = 1, then apply every local rule in
        reverse construction order.

        The ordering is found by a depth-first walk that lists parents before
        children (a topological sort); walking that list backward guarantees a
        node's gradient is complete before it hands gradient to its parents.
        """
        nodes_in_construction_order = []
        already_visited = set()

        def visit_parents_first(value):
            if id(value) not in already_visited:
                already_visited.add(id(value))
                for parent in value._parent_values:
                    visit_parents_first(parent)
                nodes_in_construction_order.append(value)

        visit_parents_first(self)
        self.gradient = 1.0
        for value in reversed(nodes_in_construction_order):
            value._propagate_gradient_to_parents()


def demonstrate_chain_rule():
    """Section 1 of the chapter: dy/dx of y = (2x+1)^2 at x = 1 equals 12."""
    def composed_function(x):
        return (2.0 * x + 1.0) ** 2

    small_step = 1e-3
    numerical_derivative = (composed_function(1.0 + small_step) - composed_function(1.0 - small_step)) / (2 * small_step)
    print("1. Chain rule check: y = (2x+1)^2 at x=1")
    print(f"   by the chain rule: dy/du * du/dx = 6 * 2 = 12")
    print(f"   numerically:       {numerical_derivative:.3f}")


def demonstrate_graph_backpropagation():
    """Sections 2-3: backpropagate L = (a*b + c)^2 and verify every gradient."""
    input_a = TrackedValue(2.0)
    input_b = TrackedValue(3.0)
    input_c = TrackedValue(-1.0)
    product_u = input_a * input_b
    sum_v = product_u + input_c
    loss_l = sum_v * sum_v
    loss_l.run_backward_pass()

    print()
    print("2. Backpropagation through L = (a*b + c)^2 with a=2, b=3, c=-1")
    print(f"   L = {loss_l.data:.0f}")
    print(f"   dL/da = {input_a.gradient:.0f}   dL/db = {input_b.gradient:.0f}   dL/dc = {input_c.gradient:.0f}   (the figure's numbers)")

    # The engine grades itself: every gradient must match the slow-but-sure
    # numerical method from Chapter 3.
    def loss_from_plain_numbers(a, b, c):
        return (a * b + c) ** 2

    small_step = 1e-6
    print("   numerical re-check:", end="")
    for name, gradient, argument_index in (("a", input_a.gradient, 0), ("b", input_b.gradient, 1), ("c", input_c.gradient, 2)):
        arguments_up = [2.0, 3.0, -1.0]
        arguments_down = [2.0, 3.0, -1.0]
        arguments_up[argument_index] += small_step
        arguments_down[argument_index] -= small_step
        numerical_gradient = (loss_from_plain_numbers(*arguments_up) - loss_from_plain_numbers(*arguments_down)) / (2 * small_step)
        print(f"  dL/d{name} = {numerical_gradient:.3f}", end="")
    print()


XOR_INPUTS = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
XOR_TARGETS = [-1.0, 1.0, 1.0, -1.0]  # tanh's natural range is (-1, 1)

# Real neural networks start their weights at small RANDOM values, never chosen
# by hand: randomness is what breaks the symmetry between neurons so they can
# specialize instead of all learning the same thing (Chapter 11 covers weight
# initialization in full). So we draw the 13 starting weights randomly here too.
# To keep the run reproducible - and identical to the C port - we seed a tiny,
# fully specified generator rather than the language's built-in random module.
WEIGHT_INITIALIZATION_SEED = 42
INITIAL_WEIGHT_RANGE = 1.0          # each weight starts uniformly random in [-1, +1]


class ReproducibleRandomGenerator:
    """A minimal linear congruential generator (LCG) for reproducible weights.

    Arguments:
        seed: the starting state; the same seed always yields the same sequence.

    Python's own random module would work, but its algorithm is intricate and
    hard to mirror in C. This LCG is a few lines of integer arithmetic, so the C
    port runs the identical sequence and starts from identical weights.
    """

    def __init__(self, seed):
        self._state = seed & 0xFFFFFFFFFFFFFFFF

    def next_uniform(self, low, high):
        """Return the next pseudo-random float, uniform in [low, high)."""
        # Advance the 64-bit state (Numerical Recipes multiplier/increment), then
        # map its top 53 bits to a double in [0, 1) and rescale to [low, high).
        self._state = (self._state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        unit_interval = (self._state >> 11) * (1.0 / 9007199254740992.0)
        return low + unit_interval * (high - low)


def train_xor_network():
    """Section 5: train the 2-3-1 tanh network on XOR with the engine.

    The training loop is Chapter 5's (forward, loss, gradients, update); the
    only novelty is that the gradient step is now one call to
    run_backward_pass() instead of hand-derived formulas.
    """
    # Draw the 13 starting weights randomly (see ReproducibleRandomGenerator), in
    # a fixed order: 3 hidden neurons, each [weight for x1, weight for x2, bias],
    # then the output neuron [three weights, bias]. The C port draws in the same
    # order from the same generator, so both start identical.
    weight_generator = ReproducibleRandomGenerator(WEIGHT_INITIALIZATION_SEED)

    def random_weight():
        return TrackedValue(weight_generator.next_uniform(-INITIAL_WEIGHT_RANGE, INITIAL_WEIGHT_RANGE))

    hidden_neuron_parameters = [[random_weight() for _ in range(3)] for _ in range(3)]
    output_neuron_parameters = [random_weight() for _ in range(4)]
    all_parameters = [parameter for neuron in hidden_neuron_parameters for parameter in neuron]
    all_parameters += output_neuron_parameters

    def network_forward(first_input, second_input):
        hidden_activations = []
        for weight_for_x1, weight_for_x2, bias in hidden_neuron_parameters:
            weighted_sum = weight_for_x1 * first_input + weight_for_x2 * second_input + bias
            hidden_activations.append(weighted_sum.tanh())
        output_weighted_sum = (
            output_neuron_parameters[0] * hidden_activations[0]
            + output_neuron_parameters[1] * hidden_activations[1]
            + output_neuron_parameters[2] * hidden_activations[2]
            + output_neuron_parameters[3]
        )
        return output_weighted_sum.tanh()

    learning_rate = 0.2
    epochs_to_print = {0, 10, 50, 100, 500, 1000, 2000}
    print()
    print("3. Training the 2-3-1 tanh network on XOR (targets -1 and +1)")
    print("   epoch   loss       predictions for (0,0) (0,1) (1,0) (1,1)")
    for epoch_number in range(2001):
        predictions_this_epoch = []
        total_loss = TrackedValue(0.0)
        for (first_input, second_input), target in zip(XOR_INPUTS, XOR_TARGETS):
            prediction = network_forward(first_input, second_input)
            predictions_this_epoch.append(prediction.data)
            prediction_error = prediction - target
            total_loss = total_loss + prediction_error * prediction_error

        for parameter in all_parameters:
            parameter.gradient = 0.0
        total_loss.run_backward_pass()
        for parameter in all_parameters:
            parameter.data -= learning_rate * parameter.gradient

        if epoch_number in epochs_to_print:
            formatted_predictions = "  ".join(f"{value:+.3f}" for value in predictions_this_epoch)
            print(f"   {epoch_number:>5}   {total_loss.data:.6f}   {formatted_predictions}")

    print("   -> XOR learned: outputs near -1, +1, +1, -1. No hand-wiring this time.")


def main():
    demonstrate_chain_rule()
    demonstrate_graph_backpropagation()
    train_xor_network()


if __name__ == "__main__":
    main()
