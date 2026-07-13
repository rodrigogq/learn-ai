# Chapter 4 — Probability basics

In this chapter you will learn the last piece of foundation math: probability. Modern AI models do not output answers — they output *probabilities of answers* ("87% cat, 13% dog"), and they are trained by a probability-based score called cross-entropy. By the end you will understand that score well enough to compute it by hand, and you will have simulated thousands of dice rolls in both languages to check the theory.

<!-- CONTENTS_START -->
## Contents

- [What you will learn](#what-you-will-learn)
- [Prerequisites](#prerequisites)
- [1. Probability and distributions](#1-probability-and-distributions)
- [2. Sampling and the law of large numbers](#2-sampling-and-the-law-of-large-numbers)
- [3. Expected value: the long-run average](#3-expected-value-the-long-run-average)
- [4. Surprise and cross-entropy — how classifiers are graded](#4-surprise-and-cross-entropy-how-classifiers-are-graded)
- [Code walkthrough](#code-walkthrough)
- [Run it](#run-it)
- [What the C version covers](#what-the-c-version-covers)
- [Exercises](#exercises)
- [Next](#next)

<!-- CONTENTS_END -->

## What you will learn

- What a probability and a probability distribution are.
- Sampling, and why averages of many random draws become predictable (the law of large numbers).
- Expected value — the "long-run average" of a random quantity.
- Surprise ($-\log p$) and **cross-entropy**, the loss function used by every classifier from Chapter 6 to Chapter 24.

## Prerequisites

- [Chapter 3](../03-derivatives-and-gradients/README.md).
- The $\sum$ (sum) notation and $\log$ from [Appendix A](../../appendices/A-math-notation/README.md).

## 1. Probability and distributions

A **probability** is a number between 0 (impossible) and 1 (certain) measuring how likely an event is. Write $P(A)$ for "the probability of event $A$". A fair die gives $P(\text{rolling a 4}) = 1/6 \approx 0.167$.

A **probability distribution** lists *every* possible outcome with its probability. The probabilities must add up to exactly 1 — something must happen. For one fair die the distribution is boring: six outcomes, each $1/6$ ("uniform"). Roll **two** dice and sum them, and structure appears:

![The probability distribution of the sum of two dice](figures/two-dice-distribution.svg)

Nobody decided 7 should be special. The shape emerges from counting: 36 equally likely pairs of dice, six of which sum to 7. Distribution shapes always come from counting or measuring — that is the whole game.

## 2. Sampling and the law of large numbers

**Sampling** means drawing an actual outcome from a distribution — rolling the die for real. One sample tells you almost nothing. Many samples reveal the distribution:

| rolls of two dice | fraction that summed to 7 | exact answer 6/36 |
|-------------------|---------------------------|-------------------|
| 100 | 0.130 | 0.1667 |
| 10,000 | 0.1653 | 0.1667 |
| 1,000,000 | 0.16669 | 0.1667 |

This is the **law of large numbers**: averages over many samples converge to the true probabilities. It is why "train on more data" is almost always good advice, and why the example programs can *verify* every claim in this chapter by brute force — simulate a million times and compare.

## 3. Expected value: the long-run average

**Why this section exists.** A single random outcome is unpredictable, and you cannot optimize something unpredictable. Expected value is the tool that squeezes a whole distribution of random outcomes down to *one stable number* — and one number is exactly what an algorithm can chase up or down. This is the bridge from "chance" to "learning": two chapters from now, the quantity a model tries to shrink (its **loss**) will itself be an expected value, so we build the idea solidly here.

The **expected value** of a random quantity $X$, written $\mathbb{E}[X]$ (read "the expectation of X"), is each possible value weighted by its probability:

$$\mathbb{E}[X] = \sum_{\text{outcomes } i} P(x_i) \cdot x_i$$

Decode the symbols: $\sum$ means "add up over all the outcomes", $P(x_i)$ is the probability of outcome number $i$, and $x_i$ is that outcome's value. So you multiply each value by how often it happens and add it all up — a **weighted average**. For one die: $\frac{1}{6}(1+2+3+4+5+6) = 3.5$. You never actually roll 3.5; expectation is not a prediction of one roll, it is the number the *average of many rolls* settles onto (Section 2's law of large numbers is what guarantees that settling).

### Why a casino cannot lose — the house edge is an expected value

Here is that idea made concrete, because it is the very same mechanism that will make training work. Take a **1-dollar bet** on a single number in American roulette. The wheel has 38 slots. If your number hits (probability 1/38) you win 35 dollars; otherwise (probability 37/38) you lose your 1 dollar. The player's expected value per 1-dollar bet is:

$$\mathbb{E}[\text{player}] = \frac{1}{38}\cdot(+35) + \frac{37}{38}\cdot(-1) = \frac{35 - 37}{38} = -\frac{2}{38} \approx -0.053$$

So every dollar bet is worth about **−5.3 cents to the player**, which is the same as **+5.3 cents to the casino**. On any *single* spin this number is invisible — the player might walk away 35 dollars richer. But now bring back Section 2: over millions of independent bets, the law of large numbers drags the *average* result onto that expected value with iron reliability. So the casino's income is not luck, it is arithmetic — roughly the number of bets times the expected value per bet. A casino that takes a million 1-dollar bets expects about **53,000 dollars** of profit (a million times 5.3 cents), and the *more* bets it processes, the *less* that figure wobbles. That is exactly why casinos love volume and enforce betting limits: volume turns a tiny per-bet expected value into near-certain revenue. The expected value **is** the business model, printed right on the felt.

Insurers, lotteries, and — the reason we are here — **loss functions all run on this identical logic.** Every loss you minimize in this course is an expected value: the average wrongness over many training examples. Making a model better means pushing that one expected number down, spin after spin.

## 4. Surprise and cross-entropy — how classifiers are graded

Here is the payoff of the chapter. We need a fair score for a model that outputs probabilities. The key ingredient is **surprise**:

$$\text{surprise of an event you gave probability } p = -\log(p)$$

**First, what is a logarithm?** You need only a couple of facts, no algebra. The natural logarithm — written $\log$ here (some books write $\ln$) — takes a positive number and answers *"how many times must I multiply $e \approx 2.718$ by itself to reach this number?"* For our purposes, forget that definition and just memorize its **shape**:

| probability $p$ | $\log(p)$ | surprise $-\log(p)$ |
|---|---|---|
| 1.0 (certain) | 0 | 0 |
| 0.5 | −0.69 | 0.69 |
| 0.1 | −2.30 | 2.30 |
| 0.01 | −4.61 | 4.61 |
| → 0 (impossible) | → −∞ | → +∞ |

Two things to read off the table. First, $\log(1) = 0$, and for any number **between 0 and 1** — which is all a probability ever is — $\log$ comes out **negative**, diving toward minus infinity as the number shrinks to zero. Second, the **minus sign** in $-\log(p)$ simply flips that whole negative column into positive numbers, so that a *smaller* probability produces a *bigger* value. That flipped quantity is what we call surprise. (The base — whether $e$, 10, or 2 — only multiplies the entire column by a constant, so any base measures surprise equally well; we use $e$ because the gradient of $\log$, which Chapter 3 will need, is cleanest for that base.)

Why is $-\log(p)$ the right shape for surprise? Because it behaves exactly like surprise should:

- You said $p = 1$ (certain) and it happened: $-\log(1) = 0$. No surprise.
- You said $p = 0.5$: $-\log(0.5) \approx 0.69$. Mild surprise.
- You said $p = 0.1$ and it happened anyway: $-\log(0.1) \approx 2.30$. Very surprised.
- You said $p = 0$ (impossible) and it happened: infinite surprise — the score never forgives absolute certainty that was wrong.

![The surprise curve minus log p](figures/surprise-curve.svg)

**Cross-entropy** is simply the model's **average surprise at the true answers**:

$$\text{cross-entropy} = \frac{1}{n} \sum_{i=1}^{n} -\log\big(p_{\text{model gave to the true answer of example } i}\big)$$

**Don't let the name scare you — you need no physics for this.** "Cross-entropy" sounds like a term from a thermodynamics exam, but here it means something you already understand, and the two words break down plainly:

- **Entropy** in this context is just a fancy word for **average surprise**. That is all. We already built "surprise" as $-\log(p)$ a few lines ago; averaging it over many examples is the "entropy" part. (The word was borrowed from physics decades ago because the math has the same form — but the *idea* you need is only "average surprise", nothing more.)
- **Cross** means the two things being compared are **different sources**: you measure the surprise of *reality's* answers using *the model's* probabilities. The model proposes the odds; reality reveals what happened; cross-entropy scores how surprised the model's odds leave it when reality speaks. If the model's probabilities matched reality perfectly, this surprise would be as low as possible; the more they disagree, the higher it climbs.

So whenever you read "cross-entropy loss" for the rest of the course, quietly translate it in your head to **"on average, how surprised was the model by the correct answers"** — a lower score means a less-surprised, better-calibrated model. No physics required.

A worked example, done by hand and verified by both programs. Two weather forecasters predict "rain probability" for 5 days; it actually rained on days 1, 2, and 5:

| day | rained? | forecaster A said | A's surprise | forecaster B said | B's surprise |
|-----|---------|-------------------|--------------|-------------------|--------------|
| 1 | yes | 0.8 | 0.22 | 0.5 | 0.69 |
| 2 | yes | 0.9 | 0.11 | 0.5 | 0.69 |
| 3 | no | 0.2 → gave "no" 0.8 | 0.22 | 0.5 | 0.69 |
| 4 | no | 0.1 → gave "no" 0.9 | 0.11 | 0.5 | 0.69 |
| 5 | yes | 0.6 | 0.51 | 0.5 | 0.69 |
| | | **cross-entropy (average)** | **0.23** | | **0.69** |

Forecaster A — confident *and right* — scores 0.23. Forecaster B, who shrugs "50/50" every day, scores 0.69. Lower is better, and cross-entropy rewards exactly what we want: **confident correctness**, punishing both wishy-washiness and confident wrongness.

**Why this matters so much.** Cross-entropy is not one loss among many — it is *the* training signal for essentially every model in this course that outputs probabilities. Logistic regression (Chapter 6), the handwritten-digit network (Chapter 9), and every language model (Chapters 20–24) are all trained by making it smaller. And "an LLM learns to predict the next word" has an exact meaning hiding inside this formula: at each position in its training text, the model produces a probability for *every* possible next token, and training nudges its weights to shrink $-\log(\text{the probability it gave to the token that actually came next})$, averaged over billions of positions. The number that trained every large language model you have ever used is the same cross-entropy you just computed by hand for two weather forecasters — only with a bigger table.

So from Chapter 6 onward, "training a classifier" means precisely this: compute cross-entropy, take its gradient (Chapter 3), step downhill (also Chapter 3). You now hold all three pieces.

## Code walkthrough

The example is `python/dice_and_distributions.py`. Its theme is *verify theory by brute force*: rather than trust a formula, the program simulates the random thing thousands of times and checks that reality matches. We will read it slowly and assume **no prior programming** (Chapter 3's walkthrough introduced `def`, `return`, and `for` loops — the same primer applies here).

### Step 1 — roll the dice a million times and count

The first recipe rolls two dice over and over and tallies how often each sum shows up:

```python
def simulate_two_dice_distribution(number_of_rolls, random_generator):
    sum_counts = {possible_sum: 0 for possible_sum in range(2, 13)}
    for _ in range(number_of_rolls):
        first_die = random_generator.randint(1, 6)
        second_die = random_generator.randint(1, 6)
        sum_counts[first_die + second_die] += 1
    return {dice_sum: count / number_of_rolls for dice_sum, count in sum_counts.items()}
```

- `sum_counts = {... : 0 ...}` builds a **dictionary** — a set of labelled boxes, one per possible sum from 2 to 12, each starting at 0. Think of eleven tally sheets.
- `random_generator.randint(1, 6)` is a computer die roll: a whole random number from 1 to 6. We roll two and add them.
- `sum_counts[first_die + second_die] += 1` finds the box for that sum and adds one tally. `+= 1` is shorthand for "take what is there and add one".
- The last line divides every tally by the number of rolls, turning raw counts into **fractions** (so they add up to 1) — the empirical distribution. Roll a million times and these fractions land right on top of the theoretical bars from Section 1.

### Step 2 — the exact answer, by counting instead of rolling

To have something to compare against, a second recipe computes the true probability directly:

```python
def exact_two_dice_probability(dice_sum):
    number_of_pairs = 6 - abs(7 - dice_sum)
    return number_of_pairs / 36.0
```

There are 36 equally likely dice pairs. The count that sum to `s` is `6 - |7 - s|` (`abs` is "distance from zero", so this peaks at 6 pairs for the sum 7 and falls to 1 pair for 2 or 12). Dividing by 36 gives the exact probability. **Step 1 guesses by rolling; Step 2 knows by counting** — the program prints them side by side, and they agree. That agreement *is* the law of large numbers, demonstrated rather than asserted.

### Step 3 — expected value by averaging

Expected value (Section 3) is just "roll a lot and take the average", which is exactly what the code does:

```python
for _ in range(number_of_samples):
    single_die_total += random_generator.randint(1, 6)
    two_dice_total += random_generator.randint(1, 6) + random_generator.randint(1, 6)
return single_die_total / number_of_samples, two_dice_total / number_of_samples
```

It keeps a running total, then divides by how many rolls it did. Run it for 100, then 10,000, then a million samples and watch the average crawl toward the exact 3.5 (one die) and 7.0 (two dice) — the long-run average becoming visible.

### Step 4 — cross-entropy, the chapter's payoff

This is the recipe that reappears, essentially unchanged, as the loss function from Chapter 6 to the LLM chapters:

```python
total_surprise = 0.0
for rain_probability, rained in zip(rain_probabilities_given, it_actually_rained):
    probability_given_to_actual_outcome = rain_probability if rained else 1.0 - rain_probability
    total_surprise += -math.log(probability_given_to_actual_outcome)
return total_surprise / len(rain_probabilities_given)
```

- `zip(...)` walks two lists **in lockstep** — each turn of the loop hands you one day's forecast *and* whether it actually rained that day, together.
- The middle line is the one subtle idea. `rain_probability if rained else 1.0 - rain_probability` reads as a sentence: *if it rained, the probability the forecaster gave to what actually happened is their rain probability; otherwise it is `1 - rain_probability`* (the probability they implicitly gave to "no rain"). Cross-entropy always scores the probability assigned to **the outcome that truly occurred**.
- `-math.log(...)` turns that probability into surprise (Section 4), we sum the surprise over all days, and the final line divides by the number of days to get the **average surprise** — which is the cross-entropy.

Feed it forecaster A's confident, mostly-right numbers and it returns 0.23; feed it forecaster B's eternal "0.5" and it returns 0.69. The code has graded them exactly as the hand table did.

### Quick reference

| Function | What it does | What to notice |
|----------|--------------|----------------|
| `simulate_two_dice_distribution(rolls, rng)` | Rolls two dice `rolls` times, returns the fraction landing on each sum. | Takes an explicit `rng` (a `random.Random`) so runs are reproducible from one seed — a habit kept all course long. |
| `exact_two_dice_probability(sum)` | The exact answer by counting: `6 − |7 − sum|` pairs out of 36. | Comparing this against the simulation is the whole point — the law of large numbers, made concrete. |
| `estimate_expected_value_by_sampling(n, rng)` | Averages many rolls to estimate the expected values (3.5 and 7.0). | Watch the estimate tighten as `n` grows from 100 to a million. |
| `compute_cross_entropy_for_rain_forecasts(probs, rained)` | **The chapter's payoff:** average surprise `−log(p)` at what actually happened. | The `if rained else 1 − p` branch: on a dry day, the probability of the *actual* outcome is `1 − p_rain`. This exact function reappears as the classifier loss from Chapter 6 on. |
| `main()` | Runs the three demos and reproduces the weather-forecaster table. | Forecaster A (confident, right) scores 0.23; B ("50/50" always) scores 0.69. Lower is better. |

**Carry forward:** `compute_cross_entropy_for_rain_forecasts` is cross-entropy. Chapters 6, 9, 20–24 all minimize a version of it — you have already written the loss that trains an LLM.

## Run it

```bash
.venv/bin/python chapters/04-probability-basics/python/dice_and_distributions.py
make -C chapters/04-probability-basics/c && ./chapters/04-probability-basics/c/build/dice_and_distributions
```

Both programs: (1) simulate a million two-dice rolls and print the empirical distribution next to the exact one, (2) verify $\mathbb{E} = 3.5$ and 7.0 by sampling, (3) score the two forecasters and reproduce the table above.

## What the C version covers

A full port. Worth reading for one detail: how a random integer 1–6 is made from `rand()`, and why the code uses a fixed seed (`srand(42)`) — so every reader's output matches the chapter exactly. Reproducible randomness (seeding) is a habit you will keep for the whole course: it is how you debug models.

## Exercises

1. By hand: what is the probability the sum of two dice is at least 10? (Count the pairs.) Verify by adding a counter to either program.
2. Compute (by hand) the cross-entropy of a forecaster C who said rain 0.99, 0.99, 0.01, 0.01, 0.01 for the five days. Day 5 rained and C gave it 0.01 — watch what one confident mistake does to the average.
3. Change the simulation to 100 rolls and run it five times (vary the seed). How much do the empirical probabilities wobble? Reconcile this with the law of large numbers.
4. The expected value of one die is 3.5. What is the expected value of "the larger of two dice"? Estimate it by simulation first; then, if you enjoy counting, verify exactly ($\approx 4.47$).

## Next

[Chapter 5 — Linear regression](../05-linear-regression/README.md)

<!-- NAV_START -->
---

[← Chapter 3: Derivatives and gradients](../03-derivatives-and-gradients/README.md) · [↑ Course index](../../README.md) · [Chapter 5: Linear regression →](../05-linear-regression/README.md)

<!-- NAV_END -->
