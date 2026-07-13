# Chapter 6 — Logistic regression

Chapter 5 predicted a *number* (a price). But most AI questions want a *category*: spam or not, cat or dog, pass or fail. In this chapter you will build your first trained classifier — and meet the exact loss (cross-entropy) and output layer (sigmoid) that every classifier in this course uses, up to and including the mini-LLM.

<!-- CONTENTS_START -->
## Contents

- [What you will learn](#what-you-will-learn)
- [Prerequisites](#prerequisites)
- [1. The problem, and why a line is not enough](#1-the-problem-and-why-a-line-is-not-enough)
- [2. The sigmoid: from weighted sum to probability](#2-the-sigmoid-from-weighted-sum-to-probability)
- [3. The loss and its gradient](#3-the-loss-and-its-gradient)
- [4. Training, step by step](#4-training-step-by-step)
- [5. The decision boundary](#5-the-decision-boundary)
- [Code walkthrough](#code-walkthrough)
- [Run it](#run-it)
- [What the C version covers](#what-the-c-version-covers)
- [Exercises](#exercises)
- [Next](#next)

<!-- CONTENTS_END -->

## What you will learn

- Why predicting a category needs a different model than predicting a number.
- The sigmoid function — how a weighted sum becomes a probability.
- Training with cross-entropy (Chapter 4's "average surprise") and gradient descent.
- Decision boundaries: turning probabilities into decisions.

## Prerequisites

- [Chapter 4](../04-probability-basics/README.md) — cross-entropy.
- [Chapter 5](../05-linear-regression/README.md) — the training loop (forward, loss, gradients, update).

## 1. The problem, and why a line is not enough

Twelve students studied for an exam; here is how long each studied and whether they passed (1) or failed (0):

| hours | 0.5 | 1 | 1.5 | 2 | 2.5 | 3 | 3.5 | 4 | 4.5 | 5 | 5.5 | 6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| passed | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 1 | 1 |

Notice the messy middle: one student passed with 3.5 hours, another failed with 4. Real data always has this — identical effort, different outcomes. That is precisely why the model should output a **probability** ("with 4 hours you pass with probability 0.66"), not a hard yes/no.

Could we just reuse Chapter 5, fitting `passed = w·hours + b`? Two things break:

1. The line's output is unbounded — it happily predicts "1.4" or "−0.2 chance of passing", which is nonsense for a probability.
2. Squared error is the wrong penalty for probabilities. Chapter 4 built the right one: **cross-entropy**, which fines confident wrongness brutally and never forgives a "0% chance" that happens. Squared error would fine a 0.99-confident mistake at most $(1-0)^2 = 1$ — barely more than a shrug.

We keep the weighted sum (it is our only tool for combining features, and a good one) but fix the output range with one extra step.

## 2. The sigmoid: from weighted sum to probability

The **sigmoid function** squashes any number into the range (0, 1):

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Read it piece by piece: $z$ is any number (for us: the weighted sum $w \cdot x + b$). $e^{-z}$ is the exponential from [Appendix A](../../appendices/A-math-notation/README.md) — always positive, huge when $z$ is very negative, near zero when $z$ is very positive. So the denominator ranges from "huge" (output near 0) down to "barely above 1" (output near 1):

![The sigmoid function squashing the number line into the interval 0 to 1](figures/sigmoid-function.svg)

Check the three marked points by hand: $\sigma(0) = 1/(1+1) = 0.5$; $\sigma(2) = 1/(1+0.135) \approx 0.88$; $\sigma(-2) \approx 0.12$. Big positive weighted sum → probability near 1; big negative → near 0; zero → the 50/50 tipping point.

Our whole model is therefore:

$$P(\text{pass}) = \sigma(w \cdot \text{hours} + b)$$

Two parameters, exactly like Chapter 5 — one weight, one bias. This model is called **logistic regression** (a historical name: it is a classifier, despite the "regression").

## 3. The loss and its gradient

Chapter 4 already built our loss. The model gives each training student a probability; cross-entropy is the **average surprise at what actually happened**:

$$L(w, b) = \frac{1}{n} \sum_{i=1}^{n} -\log\big(p_i \text{ if student } i \text{ passed, else } 1 - p_i\big) \quad\text{where } p_i = \sigma(w x_i + b)$$

To train, we need the gradients. The derivation (chain rule through the log and the sigmoid) has a famous punchline — nearly everything cancels, leaving:

$$\frac{\partial L}{\partial w} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i) x_i \qquad\qquad \frac{\partial L}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (p_i - y_i)$$

Stop and compare with Chapter 5's gradients: **identical shape** — average of (error × input), average of (error) — except the "error" is now `probability − label` instead of `prediction − truth`, and the factor 2 is gone. This is not a coincidence; sigmoid + cross-entropy were made for each other, and the same clean pattern will reappear with softmax in Chapter 9. We do not reproduce the full cancellation here (it is a satisfying exercise once you have Chapter 8's chain-rule practice), but we do not ask for faith either: **both programs verify these formulas numerically before training**, exactly like Chapter 5.

## 4. Training, step by step

The loop is unchanged — *forward, loss, gradients, update* — with learning rate 0.5 (hours are small numbers, so no scaling needed; check: feature values 0.5–6, all within one order of magnitude):

| epoch | loss | $w$ | $b$ | boundary (h) |
|-------|------|-----|-----|--------------|
| 0 | 0.6931 | 0.000 | 0.000 | — |
| 10 | 0.5438 | 0.300 | −0.755 | 2.52 |
| 100 | 0.2880 | 1.082 | −3.838 | 3.55 |
| 1000 | 0.2115 | 2.253 | −8.405 | 3.73 |
| 5000 | 0.2092 | 2.605 | −9.769 | 3.75 |

Two details worth noticing:

- The starting loss is 0.6931 — that is $-\log(0.5)$, the "always say 50/50" score of Chapter 4's forecaster B. With $w = b = 0$ the sigmoid outputs 0.5 for everyone; training begins from pure ignorance.
- The final loss is not zero and never will be: the two noisy students (passed at 3.5 h, failed at 4 h) make perfect prediction impossible. The model settles on honest probabilities instead.

## 5. The decision boundary

The model outputs probabilities; decisions come from a threshold. The natural one is 0.5, and the sigmoid crosses 0.5 exactly where the weighted sum is zero:

$$w \cdot x + b = 0 \quad\Rightarrow\quad x = -\frac{b}{w} = \frac{9.769}{2.605} \approx 3.75 \text{ hours}$$

![Data, fitted probability curve, and the decision boundary at 3.75 hours](figures/pass-probability-curve.svg)

Below ~3.75 study hours the model bets "fail", above it "pass" — but unlike Chapter 1's hand-guessed `weight > 150`, this threshold was *learned*, and the model also tells you how confident it is near the line: $P(\text{pass} \mid 3.7\text{h}) = 0.47$ — a coin flip, as the messy data deserves.

With one feature the boundary is a point on the hours axis. With two features it becomes a line in the plane; with hundreds, an invisible flat surface. What it can never be, for logistic regression, is *curved* — remember this, because it is exactly the wall Chapter 7 runs into.

## Code walkthrough

The example is `python/train_logistic_regression.py`. It is deliberately Chapter 5's program with two surgical changes — a **sigmoid** on the output and **cross-entropy** for the loss — so we will read those two closely and move fast through the parts you already know. No prior programming assumed.

### Step 1 — the sigmoid: turn a weighted sum into a probability

```python
def sigmoid(weighted_sum):
    return 1.0 / (1.0 + math.exp(-weighted_sum))
```

`math.exp(-weighted_sum)` is the number $e$ raised to the power $-z$ (Section 2). This one line is the whole reason the program *classifies* rather than predicting a number: whatever the weighted sum comes out to — −50, 0, or +50 — the result lands strictly between 0 and 1, ready to be read as a probability. Feed it 0 and you get 0.5; feed it a big positive number and you get almost 1.

### Step 2 — the loss: average surprise, with a safety rail

```python
for feature_value, true_label in zip(feature_values, true_labels):
    predicted_probability = sigmoid(weight * feature_value + bias)
    probability_of_what_happened = predicted_probability if true_label == 1 else 1.0 - predicted_probability
    clamped_probability = min(max(probability_of_what_happened, PROBABILITY_CLAMP), 1.0 - PROBABILITY_CLAMP)
    total_surprise += -math.log(clamped_probability)
return total_surprise / len(feature_values)
```

This is Chapter 4's cross-entropy, now fed by the sigmoid:

- `predicted_probability` is the model's `P(pass)` for this student.
- The middle line picks the probability the model gave to **what actually happened**: if the student passed (`true_label == 1`) that is `predicted_probability`; otherwise it is `1.0 - predicted_probability`, the leftover probability of failing. (Same `if ... else` idea as Chapter 4's weather code.)
- `min(max(..., PROBABILITY_CLAMP), 1.0 - PROBABILITY_CLAMP)` nudges that probability a hair away from exact 0 and 1. Why: the next line takes `-math.log(...)`, and `log(0)` is minus infinity, which would poison the whole average. Clamping is a **numerical safety rail** every real framework has.
- `-math.log(...)` is the surprise (Chapter 4); we sum it over students and divide by the count — the average surprise, i.e. the loss.

### Step 3 — the gradients: the exact same shape as Chapter 5

```python
for feature_value, true_label in zip(feature_values, true_labels):
    prediction_error = sigmoid(weight * feature_value + bias) - true_label
    gradient_weight += prediction_error * feature_value
    gradient_bias += prediction_error
return gradient_weight / number_of_examples, gradient_bias / number_of_examples
```

Look at this next to Chapter 5's gradient code — it is line-for-line the same, accumulating `error * feature` and `error`. Only the **definition of "error" changed**: here it is `sigmoid(...) - true_label`, the predicted probability minus the 0/1 label, instead of `prediction - true_price`. That is the famous cancellation from Section 3, made concrete: sigmoid and cross-entropy were designed together precisely so the messy chain-rule derivative collapses to this clean `probability − label`. (And as always, `verify_gradients_numerically` confirms this formula against Chapter 3's central difference *before* training trusts it.)

### Step 4 — the training loop (unchanged) plus the decision boundary

```python
for epoch_number in range(5001):
    gradient_weight, gradient_bias = compute_loss_gradients(weight, bias, STUDY_HOURS, PASSED_EXAM)
    weight = weight - learning_rate * gradient_weight
    bias = bias - learning_rate * gradient_bias
```

This is the identical *forward, loss, gradient, update* skeleton you memorized in Chapter 5 — measure the gradient, step against it, repeat. Nothing about the loop knows or cares that the model now contains a sigmoid. Alongside it, the program prints the **decision boundary**, `-bias / weight`: the study-hours value where the weighted sum is zero and the sigmoid crosses 0.5 (Section 5). Watch it settle toward 3.75 hours as training proceeds.

### Step 5 — turning probabilities into decisions

```python
for hours_studied in (2.0, 3.7, 5.0):
    pass_probability = sigmoid(weight * hours_studied + bias)
    decision = "pass" if pass_probability >= 0.5 else "fail"
```

Inference: run the trained model on new students, then threshold at 0.5 to get a hard yes/no. The 3.7-hour student comes out at `P = 0.47` → "fail", but *barely* — the model reports its own uncertainty near the boundary, which is exactly what the messy data deserves.

### Quick reference

| Function | What it does | What to notice |
|----------|--------------|----------------|
| `sigmoid(z)` | `1 / (1 + e^(−z))` — squashes the weighted sum into a probability. | Three characters of math; the whole reason this is a classifier. |
| `compute_cross_entropy_loss(w, b, x, y)` | Average surprise at the true labels (Chapter 4's loss, applied here). | The `PROBABILITY_CLAMP` (1e-12) keeps `log(0)` from becoming infinity — a safety rail every real framework has. |
| `compute_loss_gradients(w, b, x, y)` | The gradients — which cancel down to `(probability − label)`. | Compare with Chapter 5: **same shape**, `error·x` and `error`, but "error" is now `probability − label`. |
| `verify_gradients_numerically(w, b)` | Numeric check of those gradients before training. | Same discipline as Chapter 5 — the formula is confirmed, not assumed. |
| `main()` | Trains 5000 epochs, prints the loss and the **decision boundary** each step, then predicts for 2 h / 3.7 h / 5 h students. | The boundary converges to 3.75 h; the 3.7 h student sits at P = 0.47, a coin flip. |

**Carry forward:** `sigmoid` + `compute_cross_entropy_loss` are the classifier core. Chapter 9 scales the same pair to ten classes (softmax); the pattern is identical.

## Run it

```bash
.venv/bin/python chapters/06-logistic-regression/python/train_logistic_regression.py
make -C chapters/06-logistic-regression/c && ./chapters/06-logistic-regression/c/build/train_logistic_regression
```

Both programs print, in order: the numerical gradient check, the training table above, the learned boundary, and predictions for three new students (2 h, 3.7 h, 5 h of study). Identical output in both languages.

## What the C version covers

A full port. One numeric subtlety appears in the code of both languages: computing $-\log(1-p)$ when $p$ is extremely close to 1 would overflow to infinity, so the loss code clamps probabilities away from exact 0 and 1. Numerical safety rails like this are routine in real ML code — you will meet the same trick inside every framework.

## Exercises

1. By hand: with the trained model ($w=2.605$, $b=-9.769$), compute $P(\text{pass})$ for a student who studies 3 hours. (Compute the weighted sum, then the sigmoid — a calculator helps for $e$.) Check against the programs.
2. Remove the two noisy students from the dataset and retrain. What happens to the final loss, and to $w$? (When classes separate perfectly, $w$ keeps growing — the model becomes ever more confident. Watch it happen.)
3. Change the decision threshold from 0.5 to 0.9 in your head: which students get classified differently? When would a real system want a 0.9 threshold instead of 0.5? (Think of a spam filter deleting mail, or a medical screen.)
4. The gradient formulas contain `(probability − label)`. Explain in one sentence why the gradient is exactly zero for a student the model predicts perfectly.
5. Challenge: add a second feature, "hours slept the night before", inventing plausible values. Extend the model to $\sigma(w_1 x_1 + w_2 x_2 + b)$, verify your gradients numerically, and find where its decision *line* sits.

## Next

[Chapter 7 — Perceptrons and neurons](../07-perceptron-and-neurons/README.md)

<!-- NAV_START -->
---

[← Chapter 5: Linear regression](../05-linear-regression/README.md) · [↑ Course index](../../README.md) · [Chapter 7: Perceptrons and neurons →](../07-perceptron-and-neurons/README.md)

<!-- NAV_END -->
