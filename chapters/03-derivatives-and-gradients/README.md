# Chapter 3 — Derivatives and gradients

In this chapter you will learn the second (and last) piece of math that powers all of AI: the derivative. Chapter 2 gave models a way to *compute* (dot products); this chapter gives them a way to *improve* (follow slopes). By the end you will write a program that finds the bottom of a valley by feeling the ground — which is literally how neural networks learn.

## What you will learn

- What a slope is, and what a derivative is (the slope of a curve at one point).
- How to compute any derivative **numerically**, with three lines of code and no formulas.
- Partial derivatives and the gradient — slopes for functions with many inputs.
- Gradient descent: the walk-downhill algorithm behind every model in this course.

## Prerequisites

- [Chapter 2](../02-vectors-and-matrices/README.md) — vectors.
- Notation reference: [Appendix A](../../appendices/A-math-notation/README.md).

## 1. Slope: how fast something changes

A straight line's slope is rise over run: climb 2 meters for every 1 meter forward → slope 2. Walk in the other direction (descending) → slope −2. Flat ground → slope 0. That is all "slope" means: **how much the output changes per unit of input change.**

Curves are trickier — their steepness changes from place to place. So we ask for the slope **at one point**:

![The derivative as the limit of secant slopes](figures/derivative-as-slope.svg)

Pick a point $x$ and a nearby point $x + h$ (where $h$ is a small step). The slope of the straight line between them (the *secant*) is:

$$\text{slope between the two points} = \frac{f(x+h) - f(x)}{h}$$

Read it: "how much did $f$ rise, divided by how far we stepped." Now shrink $h$ toward zero. The two points merge, the secant becomes the *tangent*, and the slope settles on one number. That number is the **derivative** of $f$ at $x$, written $f'(x)$ or $\frac{df}{dx}$:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

The $\lim_{h \to 0}$ (read: "the limit as h goes to zero") is just the "shrink $h$" instruction in symbols.

### You can always compute it numerically

Formulas like "the derivative of $x^2$ is $2x$" exist (calculus classes spend months on them), but you never *need* them: just plug in a small $h$:

```python
derivative_estimate = (f(x + small_step) - f(x - small_step)) / (2 * small_step)
```

Stepping both ways (the **central difference**) is more accurate than stepping only forward. With `small_step = 1e-5`, the estimate for $f(x)=x^2$ at $x=3$ comes out 6.000000000 — matching the formula $2x = 6$. The example programs verify this and more.

This numerical trick matters for two reasons: it is how you **check** any hand-derived gradient (we will use it as a safety net in Chapter 8), and it proves derivatives are nothing mystical — just $(f(x+h)-f(x-h))/2h$ with a small $h$.

## 2. Many inputs: partial derivatives and the gradient

Real models have many knobs, not one. Take $f(x, y) = x^2 + 3y^2$ — a function with two inputs, shaped like an oval bowl. Now "the slope" needs a direction:

- The **partial derivative** $\frac{\partial f}{\partial x}$ (read: "partial f by x") is the slope if you move only along $x$, holding $y$ frozen. For our bowl: $2x$.
- $\frac{\partial f}{\partial y}$ is the slope moving only along $y$: $6y$.

The **gradient** collects every partial derivative into one vector, written $\nabla f$ (the triangle is called "nabla"):

$$\nabla f(x, y) = \left( \frac{\partial f}{\partial x}, \; \frac{\partial f}{\partial y} \right) = (2x, \; 6y)$$

The gradient has a superpower, and it is the single most important fact in this course:

> **The gradient points in the direction of steepest ascent. So its opposite, $-\nabla f$, points steepest downhill.**

At the point $(2, 1)$: $\nabla f = (4, 6)$. The bowl climbs fastest in the direction $(4,6)$; to descend fastest, step toward $(-4, -6)$.

## 3. Gradient descent: learning is walking downhill

Here is the plan that trains every model from Chapter 5 to Chapter 31:

1. Stand somewhere (start with random parameter values).
2. Feel the slope under your feet (compute the gradient).
3. Take a small step downhill (subtract a fraction of the gradient).
4. Repeat.

As one formula, applied to every parameter:

$$x_{\text{new}} = x_{\text{old}} - \eta \, \nabla f(x_{\text{old}})$$

$\eta$ ("eta") is the **learning rate** — the step size. It is the first *hyperparameter* you meet (a knob *you* choose rather than the model learning it). Too small: you crawl. Too large: you overshoot the valley and bounce out. The example programs let you feel both failure modes.

![Gradient descent stepping down an oval bowl toward the minimum](figures/gradient-descent-bowl.svg)

Running descent on the bowl from $(2, 1)$ with $\eta = 0.1$:

| step | position $(x, y)$ | height $f(x,y)$ |
|------|-------------------|------------------|
| 0 | (2.000, 1.000) | 7.000 |
| 1 | (1.600, 0.400) | 3.040 |
| 2 | (1.280, 0.160) | 1.715 |
| 5 | (0.655, 0.010) | 0.430 |
| 20 | (0.023, 0.000) | 0.001 |

It slides to the bottom $(0,0)$, fast at first and gently at the end (small slope → small steps). No formula told it where the minimum was — it *found* it by feel. **Replace "bowl" with "how wrong my model is" and this is machine learning.** That replacement is exactly Chapter 5.

## Code walkthrough

The example is `python/numerical_gradients.py`. Everything is built from one idea — the central difference — so the file is short and each function adds one layer:

| Function | What it does | What to notice |
|----------|--------------|----------------|
| `estimate_derivative(function, point, step)` | The central difference `(f(x+h) − f(x−h)) / 2h` — Section 1, in three lines. | It takes a *function* as an argument. This numeric trick works on anything, with no formula needed. |
| `estimate_gradient_of_two_variable_function(f, x, y, step)` | Two partial derivatives, each freezing one input. | A gradient is just "the derivative in each direction, collected" — that is all this does. |
| `oval_bowl_function(x, y)` | The example landscape `x² + 3y²`, minimum at (0,0). | The "3" is what makes it an *oval* bowl (steeper in y) — the source of the trouble in demo 4 and, later, Chapter 5. |
| `run_gradient_descent_on_bowl(rate, x0, y0, steps, steps_to_print)` | The walk-downhill loop: measure the gradient, step against it, repeat. | The two lines `current_x -= rate * gradient_x` are *the entire learning algorithm* — the minus sign is "go downhill". |
| `main()` | Runs four demos: numeric-vs-formula check, the gradient at (2,1), convergence at rate 0.1, and divergence at rate 0.4. | Demo 4 exploding is not a bug — it is the learning rate being too big, the lesson of the chapter. |

**Carry forward:** `run_gradient_descent_on_bowl` is the skeleton every training loop in the course fleshes out. Replace "bowl" with "how wrong the model is" and you have Chapter 5.

## Run it

```bash
.venv/bin/python chapters/03-derivatives-and-gradients/python/numerical_gradients.py
make -C chapters/03-derivatives-and-gradients/c && ./chapters/03-derivatives-and-gradients/c/build/numerical_gradients
```

Both programs (same output):

1. estimate derivatives numerically and compare them with the exact formulas (they agree to ~10 decimal places),
2. compute the gradient of the bowl at $(2,1)$ and check it equals $(4, 6)$,
3. run gradient descent and print the table above,
4. demonstrate a too-large learning rate exploding.

## What the C version covers

A full port. Note how a "function of two variables" is passed around in C: as a function pointer `double (*function)(double, double)` — the C refresher ([Appendix C](../../appendices/C-c-refresher/README.md)) mentions this as the one advanced C feature the course uses.

## Exercises

1. By hand: the derivative of $f(x) = x^2$ at $x = 5$ is 10. Verify with the central difference and $h = 0.01$ on paper: compute $(5.01^2 - 4.99^2)/0.02$.
2. Change the learning rate in either program to `0.4`. Describe what the table does now, and why (the bowl's $3y^2$ direction has slope $6y$ — steeper than the step size can handle).
3. Add the function $f(x) = e^x$ to the numerical check. Its exact derivative is famously $e^x$ itself — confirm numerically at $x = 1$.
4. Modify the descent to start at $(-3, -2)$. Does it still find $(0,0)$? Why does the starting point not matter *for this bowl*? (In Chapter 11 you will meet landscapes where it very much matters.)
5. Challenge: minimize $f(x, y) = (x-3)^2 + (y+1)^2$ with descent. Where should it converge? Confirm.

## Next

[Chapter 4 — Probability basics](../04-probability-basics/README.md)
