/*
 * Chapter 10 - a minimal tensor library in pure C: what torch.Tensor is
 * underneath.
 *
 * A tensor is three things: a flat block of numbers (the storage), a SHAPE
 * saying how to read it as a grid, and STRIDES saying how many elements to
 * skip to move one step along each dimension. Everything PyTorch does with
 * views - reshape, transpose, slicing without copying - falls out of those
 * strides. This program builds the struct and demonstrates the two big
 * consequences:
 *
 *   1. transpose is FREE: swap the shape and strides, never touch the data;
 *   2. operations can walk any view correctly by using strides, so the same
 *      matmul code works on a matrix and on its transpose.
 *
 * Build and run from the repository root:
 *     make -C chapters/10-intro-to-pytorch/c
 *     ./chapters/10-intro-to-pytorch/c/build/tensor_library
 */

#include <stdio.h>
#include <stdlib.h>

typedef struct {
    double *storage;      /* flat block of numbers, possibly shared by views */
    int row_count;
    int column_count;
    int row_stride;       /* elements to skip to move down one row */
    int column_stride;    /* elements to skip to move right one column */
} Tensor2D;

/*
 * Create a fresh 2D tensor with its own zeroed storage, laid out row-major
 * (row_stride = column_count, column_stride = 1 - Chapter 2's flat layout).
 *
 * row_count, column_count: the tensor's shape.
 */
static Tensor2D create_tensor(int row_count, int column_count) {
    Tensor2D tensor;
    tensor.storage = calloc((size_t)row_count * column_count, sizeof(double));
    if (tensor.storage == NULL) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    tensor.row_count = row_count;
    tensor.column_count = column_count;
    tensor.row_stride = column_count;
    tensor.column_stride = 1;
    return tensor;
}

/*
 * The single most important function here: where does element (row, column)
 * live? The strides answer it for ANY view of the storage.
 *
 * tensor:       the tensor (or view) being indexed
 * row, column:  the element's position in the view's coordinate system
 */
static double *element_at(const Tensor2D *tensor, int row, int column) {
    return tensor->storage + (long)row * tensor->row_stride + (long)column * tensor->column_stride;
}

/*
 * Transpose WITHOUT copying: the view shares the storage; only shape and
 * strides swap. Walking "along a row" of the transpose walks down a column
 * of the original - the strides encode that automatically.
 *
 * tensor: the tensor to view transposed.
 */
static Tensor2D transpose_view(const Tensor2D *tensor) {
    Tensor2D transposed = *tensor;
    transposed.row_count = tensor->column_count;
    transposed.column_count = tensor->row_count;
    transposed.row_stride = tensor->column_stride;
    transposed.column_stride = tensor->row_stride;
    return transposed;
}

/*
 * Matrix multiplication that works on any views, because all element access
 * goes through the strides.
 *
 * first, second: the operands; first->column_count must equal second->row_count
 *
 * Returns a fresh tensor holding first @ second.
 */
static Tensor2D matrix_multiply(const Tensor2D *first, const Tensor2D *second) {
    if (first->column_count != second->row_count) {
        fprintf(stderr, "shape mismatch: (%dx%d) @ (%dx%d)\n",
                first->row_count, first->column_count, second->row_count, second->column_count);
        exit(1);
    }
    Tensor2D product = create_tensor(first->row_count, second->column_count);
    for (int row = 0; row < product.row_count; row++) {
        for (int column = 0; column < product.column_count; column++) {
            double accumulated_sum = 0.0;
            for (int inner = 0; inner < first->column_count; inner++) {
                accumulated_sum += *element_at(first, row, inner) * *element_at(second, inner, column);
            }
            *element_at(&product, row, column) = accumulated_sum;
        }
    }
    return product;
}

/*
 * Elementwise ReLU in place, walking the view through its strides.
 *
 * tensor: the tensor (or view) to rectify.
 */
static void relu_in_place(Tensor2D *tensor) {
    for (int row = 0; row < tensor->row_count; row++) {
        for (int column = 0; column < tensor->column_count; column++) {
            double *element = element_at(tensor, row, column);
            if (*element < 0.0) {
                *element = 0.0;
            }
        }
    }
}

/*
 * Print a tensor with a label, one row per line.
 *
 * label:  text printed before the tensor
 * tensor: the tensor (or view) to print
 */
static void print_tensor(const char *label, const Tensor2D *tensor) {
    printf("%s  (shape %dx%d, strides %d,%d)\n", label,
           tensor->row_count, tensor->column_count, tensor->row_stride, tensor->column_stride);
    for (int row = 0; row < tensor->row_count; row++) {
        printf("    [");
        for (int column = 0; column < tensor->column_count; column++) {
            printf(" %6.1f", *element_at(tensor, row, column));
        }
        printf(" ]\n");
    }
}

int main(void) {
    printf("A tensor = flat storage + shape + strides. Everything else follows.\n\n");

    /* Chapter 2's worked example, through the library. */
    Tensor2D weight_matrix = create_tensor(3, 2);
    double weight_values[] = {2.0, 0.0, 1.0, 3.0, 0.0, 1.0};
    for (int i = 0; i < 6; i++) {
        weight_matrix.storage[i] = weight_values[i];
    }
    Tensor2D input_vector = create_tensor(2, 1);
    *element_at(&input_vector, 0, 0) = 4.0;
    *element_at(&input_vector, 1, 0) = 2.0;

    print_tensor("W =", &weight_matrix);
    Tensor2D product = matrix_multiply(&weight_matrix, &input_vector);
    print_tensor("W @ x =", &product);
    printf("  -> Chapter 2's numbers (8, 10, 2), via strides this time.\n\n");

    /* The stride trick: transpose without copying. */
    Tensor2D transposed_view_of_w = transpose_view(&weight_matrix);
    print_tensor("W^T (a VIEW - same storage, swapped strides) =", &transposed_view_of_w);

    printf("  Proof they share storage: set W[0][1] = 99 ...\n");
    *element_at(&weight_matrix, 0, 1) = 99.0;
    printf("  ... and W^T[1][0] reads %.1f without us touching it.\n\n",
           *element_at(&transposed_view_of_w, 1, 0));

    /* The same matmul code works on the view - strides make it walk the
     * storage in transposed order with zero special cases. */
    Tensor2D gram_matrix = matrix_multiply(&transposed_view_of_w, &weight_matrix);
    print_tensor("W^T @ W (matmul on a view, no copy ever made) =", &gram_matrix);
    printf("\n");

    Tensor2D activations = create_tensor(2, 3);
    double activation_values[] = {1.5, -2.0, 0.5, -0.1, 3.0, -4.0};
    for (int i = 0; i < 6; i++) {
        activations.storage[i] = activation_values[i];
    }
    print_tensor("Before ReLU:", &activations);
    relu_in_place(&activations);
    print_tensor("After ReLU:", &activations);

    printf("\nPyTorch adds autograd (Chapter 8), hundreds of operations, and GPU\n");
    printf("kernels - but its Tensor is this struct, grown up.\n");

    free(weight_matrix.storage);
    free(input_vector.storage);
    free(product.storage);
    free(gram_matrix.storage);
    free(activations.storage);
    return 0;
}
