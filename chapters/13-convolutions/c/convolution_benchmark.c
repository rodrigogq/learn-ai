/*
 * Chapter 13 - convolution in pure C: the figure's example, then a benchmark.
 *
 * The same algorithm as the Python version (patch times kernel, summed, at
 * every window position), compiled. Run both and compare the timings on the
 * identical 224x224 workload: compiled loops close most of the gap to
 * PyTorch's tuned kernels.
 *
 * Build and run from the repository root:
 *     make -C chapters/13-convolutions/c
 *     ./chapters/13-convolutions/c/build/convolution_benchmark
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

/*
 * Slide a kernel over an image - the whole algorithm.
 *
 * input_image:    input_height * input_width values, row-major
 * input_height, input_width:  image size (AFTER any padding by the caller)
 * kernel:         kernel_size * kernel_size weights, row-major
 * kernel_size:    kernel side length (3 for a 3x3 kernel)
 * stride:         window jump per step
 * output_map:     receives output_height * output_width values, where
 *                 output_size = (input_size - kernel_size) / stride + 1
 */
static void convolve_2d(const double *input_image, int input_height, int input_width,
                        const double *kernel, int kernel_size, int stride,
                        double *output_map) {
    int output_height = (input_height - kernel_size) / stride + 1;
    int output_width = (input_width - kernel_size) / stride + 1;

    for (int output_row = 0; output_row < output_height; output_row++) {
        for (int output_column = 0; output_column < output_width; output_column++) {
            /* One weighted sum (Chapter 0) per output pixel: the patch under
             * the window times the kernel, element by element, summed. */
            double accumulated_sum = 0.0;
            for (int kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                for (int kernel_column = 0; kernel_column < kernel_size; kernel_column++) {
                    int image_row = output_row * stride + kernel_row;
                    int image_column = output_column * stride + kernel_column;
                    accumulated_sum += input_image[image_row * input_width + image_column]
                                     * kernel[kernel_row * kernel_size + kernel_column];
                }
            }
            output_map[output_row * output_width + output_column] = accumulated_sum;
        }
    }
}

/*
 * Copy an image into a larger zero-filled buffer: zero padding.
 *
 * input_image:    height * width values
 * height, width:  the unpadded size
 * padding:        zeros added on every side
 * padded_output:  receives (height+2*padding) * (width+2*padding) values
 */
static void pad_with_zeros(const double *input_image, int height, int width, int padding,
                           double *padded_output) {
    int padded_width = width + 2 * padding;
    for (int i = 0; i < (height + 2 * padding) * padded_width; i++) {
        padded_output[i] = 0.0;
    }
    for (int row = 0; row < height; row++) {
        for (int column = 0; column < width; column++) {
            padded_output[(row + padding) * padded_width + (column + padding)] =
                input_image[row * width + column];
        }
    }
}

static const double vertical_edge_kernel[9] = {
    -1.0, 0.0, 1.0,
    -1.0, 0.0, 1.0,
    -1.0, 0.0, 1.0,
};

/* Chapter 9's deterministic generator, for a reproducible benchmark image. */
static double pseudo_random_uniform(uint64_t *state) {
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(*state >> 11) * (1.0 / 9007199254740992.0);
}

int main(void) {
    /* Part 1: the figure's worked example. */
    printf("1. The figure's example: vertical-edge kernel on the 5x5 striped image\n");
    double striped_image[25];
    for (int row = 0; row < 5; row++) {
        for (int column = 0; column < 5; column++) {
            striped_image[row * 5 + column] = (column == 2 || column == 3) ? 1.0 : 0.0;
        }
    }
    double small_output[9];
    convolve_2d(striped_image, 5, 5, vertical_edge_kernel, 3, 1, small_output);
    for (int row = 0; row < 3; row++) {
        printf("   %+.0f  %+.0f  %+.0f\n",
               small_output[row * 3], small_output[row * 3 + 1], small_output[row * 3 + 2]);
    }
    printf("   Same numbers as the figure and the Python version.\n\n");

    /* Part 2: the benchmark, matching the Python workload exactly. */
    printf("2. Benchmark: 224x224 image, 3x3 kernel, padding 1 (Python's exact workload)\n");
    const int image_size = 224;
    const int padding = 1;
    const int padded_size = image_size + 2 * padding;

    double *benchmark_image = malloc(image_size * image_size * sizeof(double));
    double *padded_image = malloc(padded_size * padded_size * sizeof(double));
    double *output_map = malloc(image_size * image_size * sizeof(double));
    if (benchmark_image == NULL || padded_image == NULL || output_map == NULL) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }

    uint64_t random_state = 7;
    for (int i = 0; i < image_size * image_size; i++) {
        benchmark_image[i] = pseudo_random_uniform(&random_state);
    }
    pad_with_zeros(benchmark_image, image_size, image_size, padding, padded_image);

    /* Repeating the convolution many times makes the clock's resolution a
     * non-issue and averages away scheduling noise. */
    const int repetitions = 200;
    clock_t start_clock = clock();
    for (int repetition = 0; repetition < repetitions; repetition++) {
        convolve_2d(padded_image, padded_size, padded_size, vertical_edge_kernel, 3, 1, output_map);
    }
    double milliseconds_per_run = 1000.0 * (double)(clock() - start_clock) / CLOCKS_PER_SEC / repetitions;

    double output_checksum = 0.0;
    for (int i = 0; i < image_size * image_size; i++) {
        output_checksum += output_map[i];
    }
    printf("   C loops (-O2): %.3f ms per convolution (checksum %.1f)\n", milliseconds_per_run, output_checksum);
    printf("   Put this number next to the Python program's two timings.\n");

    free(benchmark_image);
    free(padded_image);
    free(output_map);
    return 0;
}
