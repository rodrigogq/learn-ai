# Appendix C — C refresher

The C examples in this course use a small, deliberate subset of C11. This appendix reviews exactly that subset. If you can read this page comfortably, you can read every `.c` file in the course.

## Why C at all?

Python with NumPy and PyTorch hides the machinery: one line multiplies matrices. The C versions hide nothing — every loop, every byte of memory is yours. Seeing both builds real understanding: Python for productivity, C for truth.

## Compiling and running

Every chapter's `c/` folder has a `Makefile`. From the chapter folder:

```bash
make -C c            # compile (output goes to c/build/)
./c/build/<name>     # run
```

The Makefiles use the system compiler (`cc`) with `-Wall -Wextra -O2 -lm`: all warnings on, optimizations on, math library linked.

## The pieces the course uses

### Functions with documentation blocks

```c
/*
 * Compute the dot product of two vectors.
 *
 * first_vector:      array of vector_length numbers
 * second_vector:     array of vector_length numbers
 * vector_length:     how many elements both arrays hold
 *
 * Returns the sum of element-by-element products.
 */
double compute_dot_product(const double *first_vector, const double *second_vector, int vector_length) {
    double accumulated_sum = 0.0;
    for (int element_index = 0; element_index < vector_length; element_index++) {
        accumulated_sum += first_vector[element_index] * second_vector[element_index];
    }
    return accumulated_sum;
}
```

`const double *first_vector` means "a pointer to numbers this function promises not to modify". Arrays are passed as a pointer plus a length — C does not know how long an array is by itself.

### Heap memory: `malloc` and `free`

```c
double *weight_values = malloc(number_of_weights * sizeof(double));
if (weight_values == NULL) {
    fprintf(stderr, "out of memory\n");
    return 1;
}
/* ... use weight_values like an array ... */
free(weight_values);
```

Local arrays live on the small stack; anything big (images, weight matrices) must come from `malloc`, and every `malloc` needs exactly one `free`.

### Matrices as flat 1D arrays

C has no built-in 2D dynamic arrays, so the course stores a matrix with `row_count` rows and `column_count` columns in one flat block and computes the position by hand:

```c
/* The element at (row_index, column_index) lives at this flat offset because
 * the rows are stored one after another ("row-major" order). */
double element = matrix_values[row_index * column_count + column_index];
```

This one formula appears in almost every C file in the course. NumPy stores its arrays exactly the same way — C just makes it visible.

### Structs

```c
typedef struct {
    int row_count;
    int column_count;
    double *values;     /* row-major storage, length row_count * column_count */
} Matrix;
```

A struct groups related data. From Chapter 8 on, the C examples define small structs like `Matrix`, `Neuron`, `Layer`.

### Random numbers

```c
#include <stdlib.h>

srand(42);                                            /* fixed seed: same "random" numbers every run */
double uniform_sample = (double)rand() / RAND_MAX;    /* a number in [0, 1] */
```

A fixed seed makes runs reproducible, so the output printed in each chapter matches what you see.

### Reading and writing binary files

```c
FILE *weights_file = fopen("model_weights.bin", "rb");
fread(weight_values, sizeof(double), number_of_weights, weights_file);
fclose(weights_file);
```

The later chapters export trained weights from Python as flat binary files and load them in C exactly like this. No parsing, no libraries — just numbers on disk.

### Printing

```c
#include <stdio.h>
printf("epoch %d: loss = %.4f\n", epoch_number, loss_value);
```

`%d` for integers, `%f` for doubles (`%.4f` = 4 decimal places), `%s` for strings, `\n` for newline.

## What you do NOT need

No function pointers beyond an occasional callback, no unions, no bit manipulation, no threads, no preprocessor tricks. When a chapter needs anything beyond this page, it explains it on the spot.
