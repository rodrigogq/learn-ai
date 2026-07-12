/*
 * Chapter 3 - derivatives computed numerically, gradients, and gradient
 * descent, in pure C. Full port of the Python example.
 *
 * The one new C feature here is the function pointer: a "function of one
 * variable" is passed around as `double (*function)(double)`, which reads as
 * "a pointer to a function that takes a double and returns a double".
 *
 * Build and run from the repository root:
 *     make -C chapters/03-derivatives-and-gradients/c
 *     ./chapters/03-derivatives-and-gradients/c/build/numerical_gradients
 */

#include <math.h>
#include <stdio.h>

/*
 * Estimate the derivative of a function at a point with the central difference.
 *
 * function_of_one_variable:  pointer to the function whose slope we measure
 * point:                     where to measure the slope
 * small_step:                the h in (f(x+h) - f(x-h)) / (2h); stepping both
 *                            ways cancels most of the approximation error
 *
 * Returns the estimated slope at the point.
 */
static double estimate_derivative(double (*function_of_one_variable)(double),
                                  double point, double small_step) {
    double rise = function_of_one_variable(point + small_step)
                - function_of_one_variable(point - small_step);
    double run = 2.0 * small_step;
    return rise / run;
}

/* The chapter's example landscape: f(x, y) = x^2 + 3y^2, minimum at (0, 0). */
static double oval_bowl_function(double x_value, double y_value) {
    return x_value * x_value + 3.0 * y_value * y_value;
}

/*
 * Estimate the gradient (both partial derivatives) of f(x, y) at a point.
 * Each partial derivative is an ordinary one-variable central difference with
 * the other input frozen.
 *
 * function_of_two_variables:  pointer to the two-input function
 * x_value, y_value:           the point where the gradient is measured
 * small_step:                 the h used by each central difference
 * gradient_x_out:             receives the partial derivative in x
 * gradient_y_out:             receives the partial derivative in y
 */
static void estimate_gradient_of_two_variable_function(double (*function_of_two_variables)(double, double),
                                                       double x_value, double y_value, double small_step,
                                                       double *gradient_x_out, double *gradient_y_out) {
    *gradient_x_out = (function_of_two_variables(x_value + small_step, y_value)
                     - function_of_two_variables(x_value - small_step, y_value)) / (2.0 * small_step);
    *gradient_y_out = (function_of_two_variables(x_value, y_value + small_step)
                     - function_of_two_variables(x_value, y_value - small_step)) / (2.0 * small_step);
}

/*
 * Walk downhill on the oval bowl and print selected steps.
 *
 * learning_rate:    step size (the eta in x_new = x_old - eta * gradient)
 * starting_x/y:     where the walk begins
 * number_of_steps:  how many descent updates to run
 * steps_to_print:   array of step numbers to show
 * print_count:      how many entries steps_to_print holds
 */
static void run_gradient_descent_on_bowl(double learning_rate, double starting_x, double starting_y,
                                         int number_of_steps, const int *steps_to_print, int print_count) {
    double current_x = starting_x;
    double current_y = starting_y;
    printf("  step  position (x, y)      height f(x,y)\n");
    for (int step_number = 0; step_number <= number_of_steps; step_number++) {
        int should_print_this_step = 0;
        for (int print_index = 0; print_index < print_count; print_index++) {
            if (steps_to_print[print_index] == step_number) {
                should_print_this_step = 1;
            }
        }
        if (should_print_this_step) {
            printf("  %4d  (%7.3f, %6.3f)   %10.3f\n",
                   step_number, current_x, current_y, oval_bowl_function(current_x, current_y));
        }

        double gradient_x, gradient_y;
        estimate_gradient_of_two_variable_function(oval_bowl_function, current_x, current_y,
                                                   1e-5, &gradient_x, &gradient_y);
        /* The minus signs are the entire learning algorithm: the gradient
         * points uphill, so we step the other way. */
        current_x = current_x - learning_rate * gradient_x;
        current_y = current_y - learning_rate * gradient_y;
    }
}

static double square_function(double x_value) { return x_value * x_value; }
static double cube_function(double x_value) { return x_value * x_value * x_value; }

int main(void) {
    printf("1. Numerical derivatives vs exact formulas\n");
    printf("  f(x) = x^2      at x = 3: numerical = %.10f, exact formula = 6\n",
           estimate_derivative(square_function, 3.0, 1e-5));
    printf("  f(x) = x^3      at x = 2: numerical = %.10f, exact formula = 12\n",
           estimate_derivative(cube_function, 2.0, 1e-5));
    printf("  f(x) = sin(x)   at x = 0: numerical = %.10f, exact formula = 1\n",
           estimate_derivative(sin, 0.0, 1e-5));

    printf("\n2. Gradient of the bowl f(x,y) = x^2 + 3y^2 at (2, 1)\n");
    double gradient_x, gradient_y;
    estimate_gradient_of_two_variable_function(oval_bowl_function, 2.0, 1.0, 1e-5,
                                               &gradient_x, &gradient_y);
    printf("  numerical gradient = (%.6f, %.6f), exact = (4, 6)\n", gradient_x, gradient_y);

    printf("\n3. Gradient descent from (2, 1), learning rate 0.1\n");
    int steps_to_print_converging[] = {0, 1, 2, 5, 10, 20};
    run_gradient_descent_on_bowl(0.1, 2.0, 1.0, 20, steps_to_print_converging, 6);
    printf("  -> slides to the minimum (0, 0), fast at first, gently at the end.\n");

    printf("\n4. Same walk with learning rate 0.4 (too large for the steep y direction)\n");
    int steps_to_print_diverging[] = {0, 1, 2, 3, 10};
    run_gradient_descent_on_bowl(0.4, 2.0, 1.0, 10, steps_to_print_diverging, 5);
    printf("  -> y overshoots the valley and bounces outward: divergence.\n");
    printf("     Choosing the learning rate well is a recurring theme of this course.\n");
    return 0;
}
