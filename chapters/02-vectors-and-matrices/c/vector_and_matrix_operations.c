/*
 * Chapter 2 - vector and matrix operations from scratch in pure C.
 *
 * Full port of the Python example: dot product, matrix-vector product, and
 * matrix-matrix product, verified on the chapter's worked examples, plus a
 * timed multiplication of two 200x200 matrices.
 *
 * The one idea C shows that NumPy hides: a matrix lives in memory as a flat
 * 1D block, row after row ("row-major"). The element at (row, column) is
 * matrix_values[row_index * column_count + column_index]. NumPy stores its
 * arrays exactly the same way internally.
 *
 * Build and run from the repository root:
 *     make -C chapters/02-vectors-and-matrices/c
 *     ./chapters/02-vectors-and-matrices/c/build/vector_and_matrix_operations
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
 * Multiply two vectors element by element and sum the results.
 *
 * first_vector:   array of vector_length numbers
 * second_vector:  array of vector_length numbers
 * vector_length:  how many elements both arrays hold
 *
 * Returns the single number sum_i first_vector[i] * second_vector[i].
 */
static double compute_dot_product(const double *first_vector, const double *second_vector,
                                  int vector_length) {
    double accumulated_sum = 0.0;
    for (int element_index = 0; element_index < vector_length; element_index++) {
        accumulated_sum += first_vector[element_index] * second_vector[element_index];
    }
    return accumulated_sum;
}

/*
 * Compute the matrix-vector product: each matrix row dots the vector.
 *
 * matrix_values:  row_count * column_count numbers in row-major order
 * input_vector:   array of column_count numbers
 * row_count:      number of matrix rows
 * column_count:   number of matrix columns (= the vector's length)
 * result_vector:  receives row_count numbers, one dot product per row
 */
static void multiply_matrix_by_vector(const double *matrix_values, const double *input_vector,
                                      int row_count, int column_count, double *result_vector) {
    for (int row_index = 0; row_index < row_count; row_index++) {
        /* A row starts at row_index * column_count because rows are stored
         * one after another in the flat block. */
        const double *row_start = matrix_values + row_index * column_count;
        result_vector[row_index] = compute_dot_product(row_start, input_vector, column_count);
    }
}

/*
 * Compute the matrix-matrix product with the classic triple loop.
 *
 * first_matrix_values:   row_count x inner_size, row-major
 * second_matrix_values:  inner_size x column_count, row-major
 * row_count:             rows of the first matrix (and of the result)
 * inner_size:            columns of the first = rows of the second
 * column_count:          columns of the second matrix (and of the result)
 * product_values:        receives row_count x column_count numbers, row-major
 */
static void multiply_matrices(const double *first_matrix_values, const double *second_matrix_values,
                              int row_count, int inner_size, int column_count,
                              double *product_values) {
    for (int row_index = 0; row_index < row_count; row_index++) {
        for (int column_index = 0; column_index < column_count; column_index++) {
            double accumulated_sum = 0.0;
            for (int inner_index = 0; inner_index < inner_size; inner_index++) {
                accumulated_sum += first_matrix_values[row_index * inner_size + inner_index]
                                 * second_matrix_values[inner_index * column_count + column_index];
            }
            product_values[row_index * column_count + column_index] = accumulated_sum;
        }
    }
}

static void run_worked_examples(void) {
    printf("Worked examples (same numbers as the chapter text):\n");

    double first_vector[] = {1.0, 2.0};
    double second_vector[] = {3.0, 1.0};
    printf("  (1,2) . (3,1) = %.0f\n", compute_dot_product(first_vector, second_vector, 2));

    /* The 3x2 matrix from the chapter figure, stored as one flat block. */
    double weight_matrix_values[] = {
        2.0, 0.0,
        1.0, 3.0,
        0.0, 1.0,
    };
    double input_vector[] = {4.0, 2.0};
    double matrix_vector_result[3];
    multiply_matrix_by_vector(weight_matrix_values, input_vector, 3, 2, matrix_vector_result);
    printf("  W x = [%.0f, %.0f, %.0f]   (the figure's example)\n",
           matrix_vector_result[0], matrix_vector_result[1], matrix_vector_result[2]);

    double first_matrix_values[] = {1.0, 2.0, 3.0, 4.0};
    double second_matrix_values[] = {5.0, 6.0, 7.0, 8.0};
    double product_values[4];
    multiply_matrices(first_matrix_values, second_matrix_values, 2, 2, 2, product_values);
    printf("  [[1,2],[3,4]] @ [[5,6],[7,8]] = [[%.0f, %.0f], [%.0f, %.0f]]\n",
           product_values[0], product_values[1], product_values[2], product_values[3]);
}

/*
 * Time the multiplication of two random square matrices.
 *
 * matrix_size: width and height of the matrices. 200 matches the Python
 *              example so the reader can compare the two languages directly.
 */
static void time_square_matrix_multiplication(int matrix_size) {
    int element_count = matrix_size * matrix_size;
    double *first_matrix_values = malloc(element_count * sizeof(double));
    double *second_matrix_values = malloc(element_count * sizeof(double));
    double *product_values = malloc(element_count * sizeof(double));
    if (first_matrix_values == NULL || second_matrix_values == NULL || product_values == NULL) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }

    /* Fixed seed so every run (and every reader) sees the same numbers. */
    srand(42);
    for (int element_index = 0; element_index < element_count; element_index++) {
        first_matrix_values[element_index] = (double)rand() / RAND_MAX;
        second_matrix_values[element_index] = (double)rand() / RAND_MAX;
    }

    clock_t start_clock = clock();
    multiply_matrices(first_matrix_values, second_matrix_values,
                      matrix_size, matrix_size, matrix_size, product_values);
    double elapsed_milliseconds = 1000.0 * (double)(clock() - start_clock) / CLOCKS_PER_SEC;

    /* Summing the result and printing it makes the multiplication's output
     * observable, so the optimizer cannot delete the work we are timing. */
    double product_checksum = 0.0;
    for (int element_index = 0; element_index < element_count; element_index++) {
        product_checksum += product_values[element_index];
    }

    printf("\n");
    printf("Speed test: multiplying two %dx%d matrices\n", matrix_size, matrix_size);
    printf("  C loops (-O2): %.1f milliseconds (result checksum %.1f)\n",
           elapsed_milliseconds, product_checksum);
    printf("  Compare this with the Python example's two numbers: compiled C loops\n");
    printf("  land close to NumPy, because NumPy IS compiled C underneath.\n");

    free(first_matrix_values);
    free(second_matrix_values);
    free(product_values);
}

int main(void) {
    run_worked_examples();
    time_square_matrix_multiplication(200);
    return 0;
}
