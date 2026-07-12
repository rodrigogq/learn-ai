"""Chapter 3 - derivatives computed numerically, gradients, and gradient descent.

Four demonstrations:
  1. Numerical derivatives (central difference) match the exact calculus formulas.
  2. The gradient of the bowl f(x, y) = x^2 + 3y^2 at (2, 1) is (4, 6).
  3. Gradient descent walks to the bowl's minimum at (0, 0).
  4. A too-large learning rate makes descent explode instead of converge.

Run from the repository root:
    .venv/bin/python chapters/03-derivatives-and-gradients/python/numerical_gradients.py
"""

import math


def estimate_derivative(function_of_one_variable, point, small_step=1e-5):
    """Estimate the derivative of a function at a point with the central difference.

    Arguments:
        function_of_one_variable: any Python function taking one number and
            returning one number.
        point: where to measure the slope.
        small_step: the h in (f(x+h) - f(x-h)) / (2h). Stepping both ways
            (central difference) cancels most of the approximation error, which
            is why this beats the one-sided (f(x+h) - f(x)) / h version.

    Returns the estimated slope at the point.
    """
    rise = function_of_one_variable(point + small_step) - function_of_one_variable(point - small_step)
    run = 2 * small_step
    return rise / run


def estimate_gradient_of_two_variable_function(function_of_two_variables, x_value, y_value, small_step=1e-5):
    """Estimate the gradient (both partial derivatives) of f(x, y) at a point.

    Arguments:
        function_of_two_variables: a Python function taking (x, y) and
            returning one number.
        x_value, y_value: the point where the gradient is measured.
        small_step: the h used by each central difference.

    Returns the pair (partial derivative in x, partial derivative in y). Each
    partial is an ordinary one-variable derivative with the other input frozen.
    """
    partial_derivative_x = (
        function_of_two_variables(x_value + small_step, y_value)
        - function_of_two_variables(x_value - small_step, y_value)
    ) / (2 * small_step)
    partial_derivative_y = (
        function_of_two_variables(x_value, y_value + small_step)
        - function_of_two_variables(x_value, y_value - small_step)
    ) / (2 * small_step)
    return partial_derivative_x, partial_derivative_y


def oval_bowl_function(x_value, y_value):
    """The chapter's example landscape: f(x, y) = x^2 + 3y^2, minimum at (0, 0)."""
    return x_value ** 2 + 3 * y_value ** 2


def run_gradient_descent_on_bowl(learning_rate, starting_x, starting_y, number_of_steps, steps_to_print):
    """Walk downhill on the oval bowl and print the requested steps.

    Arguments:
        learning_rate: step size (the eta in x_new = x_old - eta * gradient).
        starting_x, starting_y: where the walk begins.
        number_of_steps: how many descent updates to run.
        steps_to_print: which step numbers to show (printing all 20 would drown
            the pattern; a few rows tell the story).
    """
    current_x, current_y = starting_x, starting_y
    print("  step  position (x, y)      height f(x,y)")
    for step_number in range(number_of_steps + 1):
        current_height = oval_bowl_function(current_x, current_y)
        if step_number in steps_to_print:
            print(f"  {step_number:>4}  ({current_x:>7.3f}, {current_y:>6.3f})   {current_height:>10.3f}")
        gradient_x, gradient_y = estimate_gradient_of_two_variable_function(
            oval_bowl_function, current_x, current_y
        )
        # The minus signs are the entire learning algorithm: the gradient
        # points uphill, so we step the other way.
        current_x = current_x - learning_rate * gradient_x
        current_y = current_y - learning_rate * gradient_y


def main():
    print("1. Numerical derivatives vs exact formulas")
    checks = [
        ("f(x) = x^2      at x = 3", lambda x: x ** 2, 3.0, 6.0),
        ("f(x) = x^3      at x = 2", lambda x: x ** 3, 2.0, 12.0),
        ("f(x) = sin(x)   at x = 0", math.sin, 0.0, 1.0),
    ]
    for description, function, point, exact_value in checks:
        estimated_value = estimate_derivative(function, point)
        print(f"  {description}: numerical = {estimated_value:.10f}, exact formula = {exact_value}")

    print()
    print("2. Gradient of the bowl f(x,y) = x^2 + 3y^2 at (2, 1)")
    gradient_x, gradient_y = estimate_gradient_of_two_variable_function(oval_bowl_function, 2.0, 1.0)
    print(f"  numerical gradient = ({gradient_x:.6f}, {gradient_y:.6f}), exact = (4, 6)")

    print()
    print("3. Gradient descent from (2, 1), learning rate 0.1")
    run_gradient_descent_on_bowl(
        learning_rate=0.1, starting_x=2.0, starting_y=1.0,
        number_of_steps=20, steps_to_print={0, 1, 2, 5, 10, 20},
    )
    print("  -> slides to the minimum (0, 0), fast at first, gently at the end.")

    print()
    print("4. Same walk with learning rate 0.4 (too large for the steep y direction)")
    run_gradient_descent_on_bowl(
        learning_rate=0.4, starting_x=2.0, starting_y=1.0,
        number_of_steps=10, steps_to_print={0, 1, 2, 3, 10},
    )
    print("  -> y overshoots the valley and bounces outward: divergence.")
    print("     Choosing the learning rate well is a recurring theme of this course.")


if __name__ == "__main__":
    main()
