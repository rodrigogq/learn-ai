# Chapter 8 — Backpropagation

Chapter 7 ended with the field's defining question: layers solve XOR, but the weights were wired by hand — *how do we compute gradients through layers automatically?* This chapter answers it. You will learn the chain rule, see that every formula is secretly a graph of tiny operations, and then build a real **automatic differentiation engine** — the same machinery inside PyTorch — in about 100 lines of Python and, yes, in C. As the payoff, your engine will *learn* the XOR weights that Chapter 7 had to guess.

This is the most important chapter of the course. Everything after it is scale.

<!-- CONTENTS_START -->
## Contents

- [What you will learn](#what-you-will-learn)
- [Prerequisites](#prerequisites)
- [1. The chain rule](#1-the-chain-rule)
- [2. Every formula is a graph](#2-every-formula-is-a-graph)
- [3. Backpropagation, by hand](#3-backpropagation-by-hand)
- [4. Building the engine](#4-building-the-engine)
- [5. The payoff: XOR, learned this time](#5-the-payoff-xor-learned-this-time)
- [Code walkthrough](#code-walkthrough)
- [Run it](#run-it)
- [What the C version covers](#what-the-c-version-covers)
- [Exercises](#exercises)
- [Next](#next)

<!-- CONTENTS_END -->

## What you will learn

- The chain rule: how derivatives pass through composed functions.
- Computation graphs: any formula as a network of tiny operations.
- The backpropagation algorithm, worked completely by hand on a small graph.
- How to build an autograd engine (a `Value` that remembers its own history).
- Training a neural network on XOR with *learned* weights — closing Chapter 7's loop.

## Prerequisites

- [Chapter 3](../03-derivatives-and-gradients/README.md) — derivatives and the numerical checker.
- [Chapter 7](../07-perceptron-and-neurons/README.md) — neurons, tanh, the XOR problem.

## 1. The chain rule

Chapter 3 taught derivatives of single functions. Real models are functions *inside* functions: a neuron's output feeds the next neuron, whose output feeds the loss. We need the derivative of a **composition**, and the rule could not be friendlier:

> If $y$ depends on $u$, and $u$ depends on $x$, then
> $$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

Read it as gears: if $y$ turns 3× as fast as $u$, and $u$ turns 2× as fast as $x$, then $y$ turns 6× as fast as $x$. Rates through a chain **multiply**.

**Worked example.** $y = (2x + 1)^2$ at $x = 1$. Name the inner part $u = 2x + 1$ (so $y = u^2$):

```
du/dx = 2                (the inner function has slope 2 everywhere)
dy/du = 2u = 2*3 = 6     (at x=1, u=3; the square's slope there is 2u)
dy/dx = 6 * 2 = 12
```

Check it the Chapter 3 way: $\frac{(2 \cdot 1.001 + 1)^2 - (2 \cdot 0.999 + 1)^2}{0.002} = 12.000$. Both programs run this exact check.

Chains longer than two links work the same — multiply all the rates. That is the whole rule, and it is the *only* calculus fact backpropagation needs.

## 2. Every formula is a graph

Take $L = (a \cdot b + c)^2$. A computer never evaluates that "all at once" — it does one tiny operation at a time: first $d = a \cdot b$, then $e = d + c$, then $L = e^2$. Drawing the steps gives a **computation graph**, and each tiny operation has a one-line local derivative:

| operation | local derivative rule |
|-----------|----------------------|
| $e = d + c$ | $\frac{\partial e}{\partial d} = 1$ and $\frac{\partial e}{\partial c} = 1$ — addition passes gradients through unchanged |
| $d = a \cdot b$ | $\frac{\partial d}{\partial a} = b$ and $\frac{\partial d}{\partial b} = a$ — each input's slope is *the other* input |
| $L = e^2$ | $\frac{dL}{de} = 2e$ |
| $t = \tanh(z)$ | $\frac{dt}{dz} = 1 - t^2$ (a gift: the slope is computable from the *output* $t$) |

## 3. Backpropagation, by hand

**Backpropagation** is the chain rule organized as a two-pass walk over the graph:

1. **Forward pass**: compute every node's value, left to right, and remember them.
2. **Backward pass**: start at the end with $\frac{dL}{dL} = 1$, then walk right to left. Each node multiplies the gradient arriving from its right by its own local rule, and hands the result to its parents.

Here is the whole thing on our example with $a=2, b=3, c=-1$:

![Computation graph with forward values and backward gradients](figures/computation-graph.svg)

Follow the red numbers right to left: $\frac{dL}{de} = 2e = 10$; the addition passes 10 unchanged to both $d$ and $c$; the multiplication turns the 10 into $b \cdot 10 = 30$ for $a$ and $a \cdot 10 = 20$ for $b$. Done — the gradient of *every* input, in one backward sweep.

Notice two things. First, no step needed anything beyond the tables above and multiplication. Second — and this is why backprop wins — one forward plus one backward pass produces **all** the gradients at once. The numerical method of Chapter 3 would need two extra evaluations of the whole formula *per parameter*; for a million-parameter network, that is two million forward passes versus backprop's single one.

One extra rule completes the algorithm: if a value feeds *two* places (say $a$ is used twice), the gradients arriving from both paths **add up**. Keep that in mind when you read the code — it is why gradients are written `+=`, never `=`.

## 4. Building the engine

Now we automate Section 3. The design, in plain words, before any code:

> Wrap every number in a small object — call it a `TrackedValue` — that stores four things: its **data**, a slot for its **gradient**, which values it was **made from** (its parents), and a tiny function that knows the **local derivative rule** of the operation that made it.
>
> Every arithmetic operation (`+`, `*`, `tanh`) does double duty: it computes the result *and* records the wiring. Run any formula on `TrackedValue`s and the computation graph of Section 2 builds itself as a side effect.
>
> Then `backward()` is Section 3, mechanized: visit the nodes in reverse construction order (parents always exist before children, so construction order is already a valid ordering of the graph), seed the final node's gradient with 1, and let each node run its local rule.

That is the entire architecture of PyTorch's autograd. Ours differs in scale (one number per node instead of a whole tensor) and speed, not in concept.

The Python version (`python/tiny_autograd.py`) spells this out in ~100 heavily documented lines. The engine checks itself: after backpropagating the Section 3 example, it reproduces the figure's gradients, then re-verifies *every* gradient with Chapter 3's numerical checker.

## 5. The payoff: XOR, learned this time

With the engine, Chapter 7's embarrassment disappears. We build a tiny network — 2 inputs → 3 tanh neurons → 1 tanh output — as ordinary arithmetic on `TrackedValue`s, and train it on XOR with targets −1 and +1 (tanh's natural range):

```
forward:   build the graph, out = network(x), loss = sum of (out - target)^2
backward:  loss.backward()            <- gradients for all 13 parameters, automatically
update:    each parameter: data -= learning_rate * gradient
```

The loop is Chapter 5's — *forward, loss, gradients, update* — with step 3 now fully automatic. Training output (identical in both languages; the 13 starting weights are fixed numbers, listed in the code, so every run matches):

```
epoch   loss       predictions for (0,0) (0,1) (1,0) (1,1)
    0   4.866806   +0.206  -0.122  +0.708  +0.439      <- random nonsense
   50   0.064431   -0.976  +0.863  +0.871  -0.832      <- shape of XOR appearing
 2000   0.000515   -0.995  +0.988  +0.988  -0.986      <- XOR, learned
```

No truth-table staring. The gradients flowed backward through two layers and found weights that Chapter 7 needed a human for. This exact mechanism — bigger, batched, on a GPU — is how the mini-LLM in Chapter 24 will learn to write.

## Code walkthrough

The example is `python/tiny_autograd.py`, and it is the most important code in the course — a working autograd engine in ~100 lines. The heart is one class:

| Piece | What it does | What to notice |
|-------|--------------|----------------|
| `class TrackedValue` | Wraps a number and remembers four things: its **data**, its **gradient**, the **parents** it was made from, and a small function holding the operation's **local derivative rule**. | This is exactly PyTorch's autograd, shrunk to one number per node. Everything else is built on it. |
| `__add__`, `__mul__`, `tanh` (methods) | Each does *double duty*: computes the result **and** records how to send gradient back to its parents. | Look at how `__mul__` sends each parent `other.data * out.grad` — the multiply rule "each input's slope is the other input". The graph builds itself as a side effect of doing arithmetic. |
| `run_backward_pass()` (method) | Seeds the final gradient with 1, then visits nodes in reverse order applying each local rule. | The `+=` on gradients (never `=`) is the "a value used twice collects gradient from both paths" rule from Section 3. |
| `demonstrate_chain_rule()` | Verifies the Section 1 example numerically. | Warms you up before the engine. |
| `demonstrate_graph_backpropagation()` | Backpropagates `L = (a·b + c)²`, reproduces the figure's gradients (30, 20, 10), **and re-checks them numerically**. | The numeric re-check is the engine grading itself against Chapter 3. |
| `train_xor_network()` | Builds a 2-3-1 tanh net out of `TrackedValue`s and trains it on XOR. | The training loop is Chapter 5's — but step 3 is now a single `loss.run_backward_pass()`. That line is the payoff of the whole chapter. |

The C version (`c/tiny_autograd.c`) does the same with an **arena** — one array of node structs, each storing its operation and parent indices — because C has no operator overloading. It is closer to how real frameworks actually work than the Python.

## Run it

```bash
.venv/bin/python chapters/08-backpropagation/python/tiny_autograd.py
make -C chapters/08-backpropagation/c && ./chapters/08-backpropagation/c/build/tiny_autograd
```

Both programs print: the chain-rule worked example, the Section 3 graph gradients (matching the figure), the numerical verification of the engine, and the XOR training table.

## What the C version covers

A full port with one structural difference worth studying. Python builds the graph with operator overloading (`a * b` on objects) and closures; C has neither, so the C engine uses an **arena**: one big array of node structs, where each node stores its operation type and the *indices* of its parents. `value_multiply(graph, a, b)` replaces `a * b`, and the backward pass is a plain reverse loop over the array with a `switch` on the operation. It is less pretty and completely transparent — you can inspect every byte of the "autograd tape". Real frameworks are much closer to the C version than to the Python one.

## Exercises

1. By hand: for $L = (a \cdot b + c)^2$ with $a=2, b=3, c=-1$, we found $\partial L/\partial a = 30$. Predict what happens to $L$ if $a$ moves from 2 to 2.01, then verify: compute $L(2.01, 3, -1)$ and compare with $25 + 30 \times 0.01$.
2. Add a `sigmoid` operation to either engine. Its local rule: the output $s$ satisfies $ds/dz = s(1-s)$. Verify your implementation with the numerical checker.
3. Draw (paper) the computation graph of $f = (a + b) \cdot (a - b)$ and backpropagate $a=3, b=1$ by hand. Careful: $a$ and $b$ each feed two nodes — remember the `+=` rule. Check with the engine.
4. In the XOR training, change the hidden layer from 3 neurons to 2 and retrain. Then try 1. Explain the result with Chapter 7's geometry (how many straight lines does XOR need?).
5. Challenge: our engine recomputes `tanh` as a single operation with a known rule. But $\tanh(z) = (e^{2z}-1)/(e^{2z}+1)$ could also be built from more primitive nodes (exp, subtract, divide). What is the trade-off? (This is a real engineering decision inside every framework — look up "fused operations" if curious.)

## Next

[Chapter 9 — Your first neural network](../09-first-neural-network/README.md)

<!-- NAV_START -->
---

[← Chapter 7: Perceptrons and neurons](../07-perceptron-and-neurons/README.md) · [↑ Course index](../../README.md) · [Chapter 9: Your first neural network →](../09-first-neural-network/README.md)

<!-- NAV_END -->
