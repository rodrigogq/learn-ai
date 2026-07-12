/*
 * Chapter 9 - a full neural network trained on MNIST, from scratch in pure C.
 *
 * Same architecture and hyperparameters as the Python version: 784 inputs ->
 * 128 hidden ReLU neurons -> 10 softmax outputs, mini-batch SGD, batch 100,
 * learning rate 0.1, 5 epochs. Accuracy lands in the same ~96% region as
 * Python, though not bit-identical: the random initialization and the
 * floating-point summation order differ (the chapter discusses this).
 *
 * The program carries its own tiny random generator instead of rand() so the
 * output is identical on every operating system and C library.
 *
 * Before the first run, export the dataset (from the repository root):
 *     .venv/bin/python chapters/09-first-neural-network/python/export_mnist_for_c.py
 *
 * Build and run from the repository root:
 *     make -C chapters/09-first-neural-network/c
 *     ./chapters/09-first-neural-network/c/build/train_mnist_mlp          (full, ~1-2 min)
 *     ./chapters/09-first-neural-network/c/build/train_mnist_mlp --quick  (seconds)
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define CLASS_COUNT 10
#define BATCH_SIZE 100

/* ------------------------------------------------- deterministic randomness */

/*
 * A small linear congruential generator. We avoid rand() because each C
 * library implements it differently; with our own generator every reader
 * sees exactly the same initial weights and shuffles.
 *
 * state: pointer to the generator's evolving 64-bit state.
 *
 * Returns a uniform double in [0, 1).
 */
static double pseudo_random_uniform(uint64_t *state) {
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    /* The high 53 bits are the well-mixed ones; 2^-53 scales them into [0,1). */
    return (double)(*state >> 11) * (1.0 / 9007199254740992.0);
}

/*
 * Sample from the standard normal distribution via the Box-Muller transform:
 * two independent uniforms become one normal sample.
 *
 * state: pointer to the random generator state.
 */
static double sample_standard_normal(uint64_t *state) {
    double first_uniform = pseudo_random_uniform(state);
    double second_uniform = pseudo_random_uniform(state);
    /* Guard against log(0). */
    if (first_uniform < 1e-300) {
        first_uniform = 1e-300;
    }
    return sqrt(-2.0 * log(first_uniform)) * cos(2.0 * 3.14159265358979323846 * second_uniform);
}

/* --------------------------------------------------------- dataset loading */

/*
 * Load a whole binary file into memory.
 *
 * file_path:       path relative to the working directory (the repo root)
 * bytes_per_item:  784 for image files, 1 for label files
 * item_count_out:  receives the number of items (file size / bytes_per_item)
 *
 * Returns a malloc'd buffer with the raw bytes; exits with a helpful message
 * if the file is missing (the usual cause: the export script was not run).
 */
static uint8_t *load_binary_file(const char *file_path, long bytes_per_item, long *item_count_out) {
    FILE *file = fopen(file_path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Cannot open %s\n", file_path);
        fprintf(stderr, "Run the export script first (from the repository root):\n");
        fprintf(stderr, "  .venv/bin/python chapters/09-first-neural-network/python/export_mnist_for_c.py\n");
        exit(1);
    }
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    uint8_t *buffer = malloc(file_size);
    if (buffer == NULL || fread(buffer, 1, file_size, file) != (size_t)file_size) {
        fprintf(stderr, "Failed to read %s\n", file_path);
        exit(1);
    }
    fclose(file);
    *item_count_out = file_size / bytes_per_item;
    return buffer;
}

/* ------------------------------------------------------------- the network */

/* All parameters in one struct; matrices are flat row-major blocks, indexed
 * with the Chapter 2 formula: row * column_count + column. */
typedef struct {
    double hidden_weights[INPUT_SIZE * HIDDEN_SIZE];
    double hidden_biases[HIDDEN_SIZE];
    double output_weights[HIDDEN_SIZE * CLASS_COUNT];
    double output_biases[CLASS_COUNT];
} Network;

/*
 * He initialization: normal weights scaled by sqrt(2 / fan_in) keep the
 * signal's variance stable through ReLU layers (Chapter 11 returns to this).
 *
 * network:      the parameter struct to fill
 * random_state: pointer to the random generator state
 */
static void initialize_network(Network *network, uint64_t *random_state) {
    double hidden_scale = sqrt(2.0 / INPUT_SIZE);
    for (int weight_index = 0; weight_index < INPUT_SIZE * HIDDEN_SIZE; weight_index++) {
        network->hidden_weights[weight_index] = hidden_scale * sample_standard_normal(random_state);
    }
    memset(network->hidden_biases, 0, sizeof(network->hidden_biases));

    double output_scale = sqrt(2.0 / HIDDEN_SIZE);
    for (int weight_index = 0; weight_index < HIDDEN_SIZE * CLASS_COUNT; weight_index++) {
        network->output_weights[weight_index] = output_scale * sample_standard_normal(random_state);
    }
    memset(network->output_biases, 0, sizeof(network->output_biases));
}

/*
 * Forward pass for one batch of images.
 *
 * network:              the parameters
 * image_pixels:         batch_count * 784 raw bytes (0-255)
 * batch_count:          how many images are in the batch
 * hidden_pre_out:       receives batch_count * HIDDEN_SIZE weighted sums
 *                       (backprop needs them for the ReLU mask)
 * hidden_activation_out: receives the same after ReLU
 * class_probability_out: receives batch_count * CLASS_COUNT softmax rows
 */
static void forward_pass(const Network *network, const uint8_t *image_pixels, int batch_count,
                         double *hidden_pre_out, double *hidden_activation_out,
                         double *class_probability_out) {
    for (int example_index = 0; example_index < batch_count; example_index++) {
        const uint8_t *pixels = image_pixels + (long)example_index * INPUT_SIZE;

        for (int hidden_index = 0; hidden_index < HIDDEN_SIZE; hidden_index++) {
            double weighted_sum = network->hidden_biases[hidden_index];
            for (int pixel_index = 0; pixel_index < INPUT_SIZE; pixel_index++) {
                /* Pixels are scaled to 0..1 here, on the fly, matching the
                 * Python version's /255 preprocessing. */
                weighted_sum += (pixels[pixel_index] / 255.0)
                              * network->hidden_weights[pixel_index * HIDDEN_SIZE + hidden_index];
            }
            hidden_pre_out[example_index * HIDDEN_SIZE + hidden_index] = weighted_sum;
            hidden_activation_out[example_index * HIDDEN_SIZE + hidden_index] =
                weighted_sum > 0.0 ? weighted_sum : 0.0;
        }

        double class_scores[CLASS_COUNT];
        double maximum_score = -1e300;
        for (int class_index = 0; class_index < CLASS_COUNT; class_index++) {
            double weighted_sum = network->output_biases[class_index];
            for (int hidden_index = 0; hidden_index < HIDDEN_SIZE; hidden_index++) {
                weighted_sum += hidden_activation_out[example_index * HIDDEN_SIZE + hidden_index]
                              * network->output_weights[hidden_index * CLASS_COUNT + class_index];
            }
            class_scores[class_index] = weighted_sum;
            if (weighted_sum > maximum_score) {
                maximum_score = weighted_sum;
            }
        }

        /* Softmax with the max subtracted first, so exp() cannot overflow. */
        double exponent_sum = 0.0;
        for (int class_index = 0; class_index < CLASS_COUNT; class_index++) {
            class_scores[class_index] = exp(class_scores[class_index] - maximum_score);
            exponent_sum += class_scores[class_index];
        }
        for (int class_index = 0; class_index < CLASS_COUNT; class_index++) {
            class_probability_out[example_index * CLASS_COUNT + class_index] =
                class_scores[class_index] / exponent_sum;
        }
    }
}

/*
 * One training step on one batch: forward, cross-entropy loss, hand-derived
 * backprop (the chapter's matrix formulas, written as loops), SGD update.
 *
 * network:       the parameters, updated in place
 * image_pixels:  BATCH_SIZE * 784 raw bytes
 * labels:        BATCH_SIZE true digits
 * learning_rate: SGD step size
 *
 * Returns the batch's average cross-entropy loss.
 */
static double train_on_batch(Network *network, const uint8_t *image_pixels,
                             const uint8_t *labels, double learning_rate) {
    static double hidden_pre[BATCH_SIZE * HIDDEN_SIZE];
    static double hidden_activation[BATCH_SIZE * HIDDEN_SIZE];
    static double class_probability[BATCH_SIZE * CLASS_COUNT];
    static double output_score_gradient[BATCH_SIZE * CLASS_COUNT];
    static double hidden_pre_gradient[BATCH_SIZE * HIDDEN_SIZE];

    forward_pass(network, image_pixels, BATCH_SIZE, hidden_pre, hidden_activation, class_probability);

    double loss_sum = 0.0;
    for (int example_index = 0; example_index < BATCH_SIZE; example_index++) {
        double probability_of_true_class = class_probability[example_index * CLASS_COUNT + labels[example_index]];
        if (probability_of_true_class < 1e-12) {
            probability_of_true_class = 1e-12;
        }
        loss_sum += -log(probability_of_true_class);

        /* Softmax + cross-entropy cancel to (probability - one_hot) / batch,
         * the same clean "error" as Chapters 5 and 6. */
        for (int class_index = 0; class_index < CLASS_COUNT; class_index++) {
            double one_hot = class_index == labels[example_index] ? 1.0 : 0.0;
            output_score_gradient[example_index * CLASS_COUNT + class_index] =
                (class_probability[example_index * CLASS_COUNT + class_index] - one_hot) / BATCH_SIZE;
        }
    }

    /* Backprop into the hidden layer: dHidden = dScores @ W2^T, then the ReLU
     * mask (gradient flows only where the weighted sum was positive). */
    for (int example_index = 0; example_index < BATCH_SIZE; example_index++) {
        for (int hidden_index = 0; hidden_index < HIDDEN_SIZE; hidden_index++) {
            double gradient = 0.0;
            for (int class_index = 0; class_index < CLASS_COUNT; class_index++) {
                gradient += output_score_gradient[example_index * CLASS_COUNT + class_index]
                          * network->output_weights[hidden_index * CLASS_COUNT + class_index];
            }
            hidden_pre_gradient[example_index * HIDDEN_SIZE + hidden_index] =
                hidden_pre[example_index * HIDDEN_SIZE + hidden_index] > 0.0 ? gradient : 0.0;
        }
    }

    /* Update the output layer: dW2 = H^T @ dScores, db2 = column sums. The
     * update is applied immediately (not stored) to keep memory small. */
    for (int hidden_index = 0; hidden_index < HIDDEN_SIZE; hidden_index++) {
        for (int class_index = 0; class_index < CLASS_COUNT; class_index++) {
            double weight_gradient = 0.0;
            for (int example_index = 0; example_index < BATCH_SIZE; example_index++) {
                weight_gradient += hidden_activation[example_index * HIDDEN_SIZE + hidden_index]
                                 * output_score_gradient[example_index * CLASS_COUNT + class_index];
            }
            network->output_weights[hidden_index * CLASS_COUNT + class_index] -= learning_rate * weight_gradient;
        }
    }
    for (int class_index = 0; class_index < CLASS_COUNT; class_index++) {
        double bias_gradient = 0.0;
        for (int example_index = 0; example_index < BATCH_SIZE; example_index++) {
            bias_gradient += output_score_gradient[example_index * CLASS_COUNT + class_index];
        }
        network->output_biases[class_index] -= learning_rate * bias_gradient;
    }

    /* Update the hidden layer: dW1 = X^T @ dHiddenPre, db1 = column sums. */
    for (int example_index = 0; example_index < BATCH_SIZE; example_index++) {
        const uint8_t *pixels = image_pixels + (long)example_index * INPUT_SIZE;
        for (int hidden_index = 0; hidden_index < HIDDEN_SIZE; hidden_index++) {
            double gradient = hidden_pre_gradient[example_index * HIDDEN_SIZE + hidden_index];
            if (gradient == 0.0) {
                continue;  /* ReLU zeroed it: the whole row of updates would be zero. */
            }
            network->hidden_biases[hidden_index] -= learning_rate * gradient;
            for (int pixel_index = 0; pixel_index < INPUT_SIZE; pixel_index++) {
                if (pixels[pixel_index] != 0) {
                    network->hidden_weights[pixel_index * HIDDEN_SIZE + hidden_index] -=
                        learning_rate * (pixels[pixel_index] / 255.0) * gradient;
                }
            }
        }
    }

    return loss_sum / BATCH_SIZE;
}

/*
 * Fraction of images whose highest-probability class is the true digit.
 *
 * network:      the trained parameters
 * image_pixels: image_count * 784 raw bytes
 * labels:       image_count true digits
 * image_count:  how many images to evaluate
 */
static double measure_accuracy(const Network *network, const uint8_t *image_pixels,
                               const uint8_t *labels, long image_count) {
    static double hidden_pre[BATCH_SIZE * HIDDEN_SIZE];
    static double hidden_activation[BATCH_SIZE * HIDDEN_SIZE];
    static double class_probability[BATCH_SIZE * CLASS_COUNT];

    long correct_count = 0;
    for (long batch_start = 0; batch_start < image_count; batch_start += BATCH_SIZE) {
        int batch_count = (int)(image_count - batch_start < BATCH_SIZE ? image_count - batch_start : BATCH_SIZE);
        forward_pass(network, image_pixels + batch_start * INPUT_SIZE, batch_count,
                     hidden_pre, hidden_activation, class_probability);
        for (int example_index = 0; example_index < batch_count; example_index++) {
            int predicted_digit = 0;
            for (int class_index = 1; class_index < CLASS_COUNT; class_index++) {
                if (class_probability[example_index * CLASS_COUNT + class_index]
                  > class_probability[example_index * CLASS_COUNT + predicted_digit]) {
                    predicted_digit = class_index;
                }
            }
            correct_count += predicted_digit == labels[batch_start + example_index];
        }
    }
    return (double)correct_count / image_count;
}

int main(int argument_count, char **argument_values) {
    int quick_mode = argument_count > 1 && strcmp(argument_values[1], "--quick") == 0;

    long training_count, test_count, label_count;
    uint8_t *training_images = load_binary_file("datasets/mnist_train_images.bin", INPUT_SIZE, &training_count);
    uint8_t *training_labels = load_binary_file("datasets/mnist_train_labels.bin", 1, &label_count);
    uint8_t *test_images = load_binary_file("datasets/mnist_test_images.bin", INPUT_SIZE, &test_count);
    uint8_t *test_labels = load_binary_file("datasets/mnist_test_labels.bin", 1, &label_count);

    int number_of_epochs = 5;
    if (quick_mode) {
        training_count = 2000;
        test_count = 1000;
        number_of_epochs = 1;
        printf("(quick mode: 2,000 training images, 1 epoch)\n");
    }
    printf("%ld training images, %ld test images\n\n", training_count, test_count);

    uint64_t random_state = 42;
    Network *network = malloc(sizeof(Network));
    if (network == NULL) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    initialize_network(network, &random_state);

    long *shuffled_order = malloc(training_count * sizeof(long));
    for (long i = 0; i < training_count; i++) {
        shuffled_order[i] = i;
    }
    /* One batch's worth of gathered images/labels; gathering by shuffled
     * index gives the same effect as NumPy's fancy indexing. */
    static uint8_t batch_pixels[BATCH_SIZE * INPUT_SIZE];
    static uint8_t batch_labels[BATCH_SIZE];

    const double learning_rate = 0.1;
    printf("Training: batch size %d, learning rate %.1f\n", BATCH_SIZE, learning_rate);
    printf("  epoch   average loss   test accuracy   seconds\n");
    for (int epoch_number = 1; epoch_number <= number_of_epochs; epoch_number++) {
        clock_t epoch_start = clock();

        /* Fisher-Yates shuffle with our deterministic generator. */
        for (long i = training_count - 1; i > 0; i--) {
            long j = (long)(pseudo_random_uniform(&random_state) * (i + 1));
            long swap = shuffled_order[i];
            shuffled_order[i] = shuffled_order[j];
            shuffled_order[j] = swap;
        }

        double loss_sum = 0.0;
        long batch_count_total = 0;
        for (long batch_start = 0; batch_start + BATCH_SIZE <= training_count; batch_start += BATCH_SIZE) {
            for (int i = 0; i < BATCH_SIZE; i++) {
                long source_index = shuffled_order[batch_start + i];
                memcpy(batch_pixels + (long)i * INPUT_SIZE,
                       training_images + source_index * INPUT_SIZE, INPUT_SIZE);
                batch_labels[i] = training_labels[source_index];
            }
            loss_sum += train_on_batch(network, batch_pixels, batch_labels, learning_rate);
            batch_count_total++;
        }

        double epoch_seconds = (double)(clock() - epoch_start) / CLOCKS_PER_SEC;
        double test_accuracy = measure_accuracy(network, test_images, test_labels, test_count);
        printf("  %5d   %12.4f   %11.2f%%   %7.1f\n",
               epoch_number, loss_sum / batch_count_total, 100.0 * test_accuracy, epoch_seconds);
    }

    printf("\nFinal test accuracy: %.2f%% on %ld digits the network never saw.\n",
           100.0 * measure_accuracy(network, test_images, test_labels, test_count), test_count);

    free(training_images);
    free(training_labels);
    free(test_images);
    free(test_labels);
    free(network);
    free(shuffled_order);
    return 0;
}
