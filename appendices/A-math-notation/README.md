# Appendix A — Math notation, in plain English

Every math symbol used in this course, in one place. Chapters link here the first time they use a symbol. If you ever see notation you do not recognize, look it up here.

This appendix grows as the course grows.

## Numbers and variables

| Notation | Read it as | Meaning |
|----------|-----------|---------|
| $x$ | "x" | A single number we are talking about. Italic letters are just names for numbers, like variables in code. |
| $\hat{y}$ | "y hat" | A **prediction** of $y$. The hat means "our model's guess", as opposed to the true value $y$. |
| $x_i$ | "x sub i" | The $i$-th element of a list. If $x = (4, 7, 9)$ then $x_1 = 4$, $x_2 = 7$, $x_3 = 9$. Exactly `x[i]` in code (math usually counts from 1, code from 0). |
| $\approx$ | "approximately equals" | The two sides are close but not exactly equal. |
| $\propto$ | "proportional to" | Equal up to multiplication by some constant we do not care about. |

## Sums and products

| Notation | Read it as | Meaning |
|----------|-----------|---------|
| $\sum_{i=1}^{n} x_i$ | "the sum of x sub i, for i from 1 to n" | Add up all the elements: $x_1 + x_2 + \dots + x_n$. In code: `total = 0; for i in range(n): total += x[i]`. |
| $\prod_{i=1}^{n} x_i$ | "the product of x sub i, for i from 1 to n" | Multiply all the elements: $x_1 \cdot x_2 \cdot \dots \cdot x_n$. Same loop with `*=`. |
| $\frac{1}{n}\sum_{i=1}^{n} x_i$ | "the average of the x's" | Add them all, divide by how many there are. The **mean**. |

## Vectors and matrices (Chapter 2)

| Notation | Read it as | Meaning |
|----------|-----------|---------|
| $\mathbf{x}$ | "the vector x" | A list of numbers, written in bold. In code: a 1-dimensional array. |
| $\mathbf{x} \in \mathbb{R}^n$ | "x is a vector of n real numbers" | $\mathbb{R}$ is the set of all real numbers (any decimal). $\mathbb{R}^n$ means "a list of $n$ of them". So $\mathbf{x} \in \mathbb{R}^3$ is a list of 3 numbers. |
| $W$ | "the matrix W" | A grid of numbers, written as a capital letter. In code: a 2-dimensional array. |
| $W \in \mathbb{R}^{m \times n}$ | "W is an m-by-n matrix" | A grid with $m$ rows and $n$ columns. |
| $W_{ij}$ | "W sub i j" | The number in row $i$, column $j$ of the matrix. In code: `W[i][j]`. |
| $\mathbf{a} \cdot \mathbf{b}$ | "a dot b" | The **dot product**: multiply the vectors element by element, then add everything up. One single number comes out. |
| $W\mathbf{x}$ | "W times x" | Matrix–vector multiplication: each row of $W$ takes a dot product with $\mathbf{x}$. See Chapter 2. |
| $W^\top$ | "W transpose" | The matrix flipped over its diagonal: rows become columns. |

## Functions

| Notation | Read it as | Meaning |
|----------|-----------|---------|
| $f(x)$ | "f of x" | A function named $f$ applied to input $x$ — exactly a function call `f(x)` in code. |
| $f: \mathbb{R}^n \to \mathbb{R}$ | "f maps n numbers to one number" | The function's "type signature": it takes a vector of $n$ numbers and returns a single number. |
| $e^x$ or $\exp(x)$ | "e to the x" | The exponential function. $e \approx 2.718$. In code: `math.exp(x)`. It turns any number into a positive number, and it grows fast. |
| $\log(x)$ or $\ln(x)$ | "log of x" | The natural logarithm, the inverse of $e^x$: if $e^a = b$ then $\log(b) = a$. In this course $\log$ always means natural log (`math.log`). |
| $\|\mathbf{x}\|$ | "the norm of x" | The length of a vector: $\sqrt{x_1^2 + x_2^2 + \dots}$. Distance from the origin. |
| $\arg\max_x f(x)$ | "the x that maximizes f" | Not the maximum value itself, but *which input* produces it. In code: `max(range(n), key=f)`. |

## Derivatives and gradients (Chapter 3)

| Notation | Read it as | Meaning |
|----------|-----------|---------|
| $\frac{df}{dx}$ or $f'(x)$ | "the derivative of f with respect to x" | How fast $f$ changes when $x$ changes a tiny bit. The slope of $f$ at the point $x$. |
| $\frac{\partial f}{\partial x}$ | "the partial derivative of f with respect to x" | Same idea when $f$ has several inputs: the slope in the $x$ direction only, holding all other inputs still. |
| $\nabla f$ | "the gradient of f" (nabla) | All the partial derivatives collected into one vector: the direction of steepest increase. Learning happens by stepping the other way. |
| $\eta$ | "eta" | The **learning rate**: how big a step gradient descent takes. A small positive number like 0.01. |

## Probability (Chapter 4)

| Notation | Read it as | Meaning |
|----------|-----------|---------|
| $P(A)$ | "the probability of A" | A number between 0 (impossible) and 1 (certain). |
| $P(A \mid B)$ | "the probability of A given B" | The probability of $A$ once we already know $B$ happened. |
| $\mathbb{E}[X]$ | "the expected value of X" | The long-run average of a random quantity: each possible value times its probability, summed. |
| $X \sim \mathcal{N}(\mu, \sigma^2)$ | "X follows a normal distribution" | $X$ is random with the bell-curve distribution centered at $\mu$ ("mu", the mean) with spread $\sigma$ ("sigma", the standard deviation). |

## Greek letters used in this course

| Letter | Name | Typical meaning here |
|--------|------|---------------------|
| $\alpha$ | alpha | a coefficient or mixing factor |
| $\eta$ | eta | learning rate |
| $\theta$ | theta | "all the parameters of the model" bundled into one symbol |
| $\mu$ | mu | mean (average) |
| $\sigma$ | sigma | standard deviation; also the sigmoid function $\sigma(x)$ (context makes it clear) |
| $\epsilon$ | epsilon | a very small number |
| $\lambda$ | lambda | a strength knob, e.g. for regularization |
