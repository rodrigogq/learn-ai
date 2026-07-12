# Chapter 2 — Vectors and matrices

In this chapter you will learn the small amount of linear algebra that powers *all* of AI. This is not an exaggeration: a neural network spends almost 100% of its time doing exactly the two operations you will build here by hand — the dot product and matrix multiplication.

## What you will learn

- What a vector is (two views: list of numbers, arrow in space).
- The dot product — the "how similar are these two things?" operation.
- What a matrix is and how matrix–vector and matrix–matrix multiplication work.
- The shape rule that lets you predict whether any multiplication is legal.
- Why NumPy exists: the same math, a thousand times faster.

## Prerequisites

- [Chapter 1](../01-what-is-ai/README.md) — the vocabulary (features, model, parameters).
- Notation used here is collected in [Appendix A](../../appendices/A-math-notation/README.md).

## 1. Vectors: a list of numbers with a meaning

A **vector** is simply an ordered list of numbers. That is all. We write vectors in bold, like $\mathbf{x}$, and in code they are arrays:

![A vector shown both as a list of numbers and as an arrow on a grid](figures/vector-as-arrow.svg)

The fruit from Chapter 1 was already a vector: `(150.0, 0.45)` — weight and smoothness. A 28×28 pixel image of a digit is a vector of 784 brightness values. **Turning a real-world thing into a vector of features is always step one of machine learning.**

The number of elements is the vector's **dimension**. The notation $\mathbf{x} \in \mathbb{R}^n$ (read: "x is a vector of $n$ real numbers") just says the list has $n$ entries.

### Operations you can do with vectors

Say $\mathbf{a} = (1, 2)$ and $\mathbf{b} = (3, 1)$.

| Operation | Definition | Example | Meaning as arrows |
|-----------|-----------|---------|-------------------|
| addition | add element by element | $\mathbf{a}+\mathbf{b} = (4, 3)$ | put the arrows head to tail |
| scaling | multiply every element by one number | $2\mathbf{a} = (2, 4)$ | stretch the arrow |
| **dot product** | multiply element by element, **then add it all up** | $\mathbf{a}\cdot\mathbf{b} = 1{\cdot}3 + 2{\cdot}1 = 5$ | see below |

## 2. The dot product — the most important operation in this course

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i \, b_i$$

Read it aloud: "multiply the first elements together, the second elements together, and so on, then add everything up." The result is **one single number**. In code:

```python
dot_product = 0.0
for element_index in range(vector_length):
    dot_product += first_vector[element_index] * second_vector[element_index]
```

Why does AI care so much? Two reasons:

1. **A dot product is a weighted sum.** Chapter 0's C program combined values and importance weights: `0.5·0.8 + 0.3·(−0.2)`. Written with vectors, that is exactly `values · weights` — the dot product *is* the weighted sum from Chapter 0, in compact notation. Since weighted sums are the operation AI is built from, the dot product is the operation AI is built from. (In Chapter 7 this same computation, plus an offset, becomes the building block of neural networks.)
2. **It measures agreement.** When the vectors point the same way the dot product is large and positive; perpendicular gives zero; opposite gives negative. So `features · weights` asks: *how much does this input look like the pattern stored in the weights?* That single idea scales from spam filters up to the attention mechanism inside LLMs (Chapter 22 is one big dot product festival).

A vector's **length** (norm) is $\|\mathbf{x}\| = \sqrt{\mathbf{x} \cdot \mathbf{x}}$ — the dot product of a vector with itself, square-rooted. For $(3,2)$: $\sqrt{9+4} = \sqrt{13} \approx 3.61$. The distance you computed in Chapter 1's fruit classifier was the length of the difference between two vectors.

## 3. Matrices: many dot products at once

A **matrix** is a grid of numbers with $m$ rows and $n$ columns, written $W \in \mathbb{R}^{m \times n}$. The element in row $i$, column $j$ is $W_{ij}$ — in code, `W[i][j]`.

**Matrix × vector**: each row of the matrix takes a dot product with the vector.

![Matrix times vector worked out element by element](figures/matrix-vector-product.svg)

Why this matters: a **layer** of a neural network is $m$ neurons all looking at the same $n$ inputs. Stack each neuron's weights as one row of a matrix and the whole layer becomes a single matrix–vector product $W\mathbf{x}$ — one line of math, one line of code, and (Chapter 10 onward) one GPU operation.

**Matrix × matrix**: the same idea repeated. $C = AB$ means every element $C_{ij}$ is the dot product of **row $i$ of $A$** with **column $j$ of $B$**:

$$C_{ij} = \sum_{k} A_{ik} \, B_{kj}$$

That triple loop over $i$, $j$, $k$ is what you will write in both languages today, and it is what GPUs were built to do billions of times per second.

### The shape rule

$$(m \times n) \cdot (n \times p) \rightarrow (m \times p)$$

The inner sizes must match, and they disappear. A $(3\times2)$ matrix times a $(2\times4)$ matrix gives $(3\times4)$. A $(3\times2)$ times a $(3\times2)$ is **illegal** — the inner sizes are 2 and 3. When your PyTorch code crashes with a "shape mismatch" error in Chapter 10 (it will), this rule is how you debug it.

## 4. Why NumPy

Python loops are slow: each pass through a loop pays Python's bookkeeping costs. NumPy stores numbers in one solid block of memory (exactly like a C array — see the C notes below) and runs loops in compiled C internally:

```python
import numpy

matrix_product = first_matrix @ second_matrix    # "@" is matrix multiplication
```

The Python example times your hand-written matmul against NumPy's `@` on 200×200 matrices. Expect NumPy to win by a factor of several hundred. That gap is why all "from scratch" chapters still use NumPy for storage and speed — we implement the *ideas* ourselves, but let NumPy run the arithmetic.

## Run it

```bash
.venv/bin/python chapters/02-vectors-and-matrices/python/vector_and_matrix_operations.py
make -C chapters/02-vectors-and-matrices/c && ./chapters/02-vectors-and-matrices/c/build/vector_and_matrix_operations
```

Both print the same worked examples (dot product = 5, the matrix–vector product `(8, 10, 2)` from the figure, a 2×2 matmul), and each ends with a speed comparison: Python-loops vs NumPy, and in C, the same multiplication so you can compare machines.

## What the C version covers

A full port. It also shows the one C idea NumPy hides: a matrix is stored as a **flat 1D block** in row-major order, and `matrix[row][column]` becomes `matrix_values[row_index * column_count + column_index]`. NumPy stores its arrays exactly this way internally — C just refuses to hide it. This formula returns in every later C example.

## Exercises

1. By hand (paper!): compute $(2, -1, 3) \cdot (1, 4, 2)$ and check yourself with either program. *(Answer below the last exercise.)*
2. By hand: what is the shape of $(4\times3)(3\times2)$? And $(3\times2)(4\times3)$? One of them is illegal — which, and why?
3. In the Python file, make the matrices 400×400 instead of 200×200. Loops get ~8× slower, NumPy barely moves. Why 8×? (Hint: the triple loop runs $n^3$ times.)
4. Write (in either language) `compute_vector_length(vector)` using the dot product, and verify $\|(3,4)\| = 5$.
5. Challenge: using the shape rule, explain why the *order* of multiplication matters: $AB \neq BA$ in general, even when both are legal.

*Answer to 1: $2·1 + (-1)·4 + 3·2 = 4$.*

## Next

[Chapter 3 — Derivatives and gradients](../03-derivatives-and-gradients/README.md)
