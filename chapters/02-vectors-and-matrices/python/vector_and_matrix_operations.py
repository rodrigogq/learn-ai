"""Chapter 2 - vector and matrix operations built from scratch, then vs NumPy.

Implements the dot product, matrix-vector product, and matrix-matrix product
with plain Python lists and loops, verifies them on the chapter's worked
examples, checks the results against NumPy, and finally times hand-written
loops against NumPy so the speed gap is concrete.

Run from the repository root:
    .venv/bin/python chapters/02-vectors-and-matrices/python/vector_and_matrix_operations.py
"""

import time

import numpy


def compute_dot_product(first_vector, second_vector):
    """Multiply two vectors element by element and sum the results.

    Arguments:
        first_vector: list of numbers.
        second_vector: list of numbers, same length as first_vector.

    Returns one number: sum over i of first_vector[i] * second_vector[i].
    """
    if len(first_vector) != len(second_vector):
        raise ValueError(f"dot product needs equal lengths, got {len(first_vector)} and {len(second_vector)}")
    accumulated_sum = 0.0
    for element_index in range(len(first_vector)):
        accumulated_sum += first_vector[element_index] * second_vector[element_index]
    return accumulated_sum


def multiply_matrix_by_vector(matrix_rows, input_vector):
    """Compute the matrix-vector product: each row dots the vector.

    Arguments:
        matrix_rows: list of rows, each row a list of numbers (all same length).
        input_vector: list of numbers, length equal to the row length.

    Returns a list with one number per matrix row.
    """
    return [compute_dot_product(row, input_vector) for row in matrix_rows]


def multiply_matrices(first_matrix_rows, second_matrix_rows):
    """Compute the matrix-matrix product with the classic triple loop.

    Arguments:
        first_matrix_rows: matrix of shape (row_count, inner_size) as list of rows.
        second_matrix_rows: matrix of shape (inner_size, column_count) as list of rows.

    Returns the product matrix of shape (row_count, column_count), where each
    element [i][j] is the dot product of row i of the first matrix with
    column j of the second.
    """
    row_count = len(first_matrix_rows)
    inner_size = len(second_matrix_rows)
    column_count = len(second_matrix_rows[0])
    if len(first_matrix_rows[0]) != inner_size:
        raise ValueError(f"shape mismatch: ({row_count}x{len(first_matrix_rows[0])}) times ({inner_size}x{column_count})")

    product_rows = [[0.0] * column_count for _ in range(row_count)]
    for row_index in range(row_count):
        for column_index in range(column_count):
            for inner_index in range(inner_size):
                product_rows[row_index][column_index] += (
                    first_matrix_rows[row_index][inner_index] * second_matrix_rows[inner_index][column_index]
                )
    return product_rows


def run_worked_examples():
    """Reproduce the chapter's hand-computed examples and verify with NumPy."""
    print("Worked examples (same numbers as the chapter text):")

    first_vector = [1.0, 2.0]
    second_vector = [3.0, 1.0]
    dot_product_result = compute_dot_product(first_vector, second_vector)
    print(f"  (1,2) . (3,1) = {dot_product_result:.0f}")

    weight_matrix = [[2.0, 0.0], [1.0, 3.0], [0.0, 1.0]]
    input_vector = [4.0, 2.0]
    matrix_vector_result = multiply_matrix_by_vector(weight_matrix, input_vector)
    print(f"  W x = {[f'{value:.0f}' for value in matrix_vector_result]}   (the figure's example)")

    first_matrix = [[1.0, 2.0], [3.0, 4.0]]
    second_matrix = [[5.0, 6.0], [7.0, 8.0]]
    matrix_product = multiply_matrices(first_matrix, second_matrix)
    print(f"  [[1,2],[3,4]] @ [[5,6],[7,8]] = {[[f'{v:.0f}' for v in row] for row in matrix_product]}")

    # NumPy is the referee: if our loops disagree with it, our understanding
    # is wrong somewhere, so this assert is the real test of the chapter.
    numpy_product = numpy.array(first_matrix) @ numpy.array(second_matrix)
    assert numpy.allclose(numpy.array(matrix_product), numpy_product)
    assert numpy.allclose(
        numpy.array(matrix_vector_result), numpy.array(weight_matrix) @ numpy.array(input_vector)
    )
    print("  NumPy agrees with all hand-written results.")


def compare_speed_against_numpy(matrix_size=200):
    """Time the hand-written matmul against NumPy on square matrices.

    Arguments:
        matrix_size: width and height of the random square matrices. 200 keeps
            the pure-Python triple loop around a second on a typical machine.
    """
    random_generator = numpy.random.default_rng(seed=42)
    first_matrix_numpy = random_generator.random((matrix_size, matrix_size))
    second_matrix_numpy = random_generator.random((matrix_size, matrix_size))
    first_matrix_lists = first_matrix_numpy.tolist()
    second_matrix_lists = second_matrix_numpy.tolist()

    print()
    print(f"Speed test: multiplying two {matrix_size}x{matrix_size} matrices")

    start_time = time.perf_counter()
    hand_written_result = multiply_matrices(first_matrix_lists, second_matrix_lists)
    hand_written_seconds = time.perf_counter() - start_time
    print(f"  Python loops: {hand_written_seconds:.3f} seconds")

    start_time = time.perf_counter()
    numpy_result = first_matrix_numpy @ second_matrix_numpy
    numpy_seconds = time.perf_counter() - start_time
    print(f"  NumPy @:      {numpy_seconds:.6f} seconds")

    assert numpy.allclose(numpy.array(hand_written_result), numpy_result)
    speed_ratio = hand_written_seconds / max(numpy_seconds, 1e-9)
    print(f"  NumPy is about {speed_ratio:.0f}x faster - same math, compiled loops.")


def main():
    run_worked_examples()
    compare_speed_against_numpy()


if __name__ == "__main__":
    main()
