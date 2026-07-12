/*
 * Chapter 12 - a data loader in pure C: what DataLoader does every epoch.
 *
 * PyTorch's DataLoader performs three jobs: SHUFFLE the example order,
 * GATHER examples into fixed-size batches, and NORMALIZE/transform them on
 * the way out. This program does all three on the MNIST files exported in
 * Chapter 9, then proves the two properties that make a loader correct:
 *
 *   1. every epoch visits every example exactly once (no loss, no repeats),
 *   2. shuffling makes each batch a fair random sample - the label mix of
 *      any batch resembles the whole dataset's mix.
 *
 * Before the first run, export the dataset (from the repository root):
 *     .venv/bin/python chapters/09-first-neural-network/python/export_mnist_for_c.py
 *
 * Build and run from the repository root:
 *     make -C chapters/12-data-pipelines/c
 *     ./chapters/12-data-pipelines/c/build/batch_loader
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IMAGE_SIZE 784
#define BATCH_SIZE 100

/* Same deterministic generator as Chapter 9, for identical output everywhere. */
static double pseudo_random_uniform(uint64_t *state) {
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(*state >> 11) * (1.0 / 9007199254740992.0);
}

/*
 * Load a whole binary file into memory (Chapter 9's loader, unchanged).
 *
 * file_path:       path relative to the repository root
 * bytes_per_item:  784 for images, 1 for labels
 * item_count_out:  receives the number of items in the file
 */
static uint8_t *load_binary_file(const char *file_path, long bytes_per_item, long *item_count_out) {
    FILE *file = fopen(file_path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Cannot open %s - run the Chapter 9 export script first.\n", file_path);
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

/* The loader's state: the dataset plus this epoch's visiting order. */
typedef struct {
    const uint8_t *image_pixels;   /* item_count * 784 raw bytes */
    const uint8_t *labels;         /* item_count raw labels */
    long item_count;
    long *visiting_order;          /* permutation of 0..item_count-1 */
    long next_position;            /* how far into the epoch we are */
} BatchLoader;

/*
 * Start a new epoch: reshuffle the visiting order (Fisher-Yates) and rewind.
 *
 * loader:        the loader to reset
 * random_state:  generator state used for the shuffle
 */
static void begin_epoch(BatchLoader *loader, uint64_t *random_state) {
    for (long i = loader->item_count - 1; i > 0; i--) {
        long j = (long)(pseudo_random_uniform(random_state) * (i + 1));
        long swap = loader->visiting_order[i];
        loader->visiting_order[i] = loader->visiting_order[j];
        loader->visiting_order[j] = swap;
    }
    loader->next_position = 0;
}

/*
 * Produce the next batch: gather pixels (normalized to 0..1 doubles, the
 * "transform" step) and labels in shuffled order.
 *
 * loader:            the loader to draw from
 * batch_pixels_out:  receives BATCH_SIZE * 784 normalized doubles
 * batch_labels_out:  receives BATCH_SIZE labels
 *
 * Returns the number of examples delivered: BATCH_SIZE normally, less on the
 * final partial batch, 0 when the epoch is exhausted.
 */
static int next_batch(BatchLoader *loader, double *batch_pixels_out, uint8_t *batch_labels_out) {
    long remaining = loader->item_count - loader->next_position;
    int batch_count = remaining < BATCH_SIZE ? (int)remaining : BATCH_SIZE;
    for (int i = 0; i < batch_count; i++) {
        long source_index = loader->visiting_order[loader->next_position + i];
        const uint8_t *source_pixels = loader->image_pixels + source_index * IMAGE_SIZE;
        for (int pixel_index = 0; pixel_index < IMAGE_SIZE; pixel_index++) {
            batch_pixels_out[(long)i * IMAGE_SIZE + pixel_index] = source_pixels[pixel_index] / 255.0;
        }
        batch_labels_out[i] = loader->labels[source_index];
    }
    loader->next_position += batch_count;
    return batch_count;
}

int main(void) {
    long image_count, label_count;
    uint8_t *image_pixels = load_binary_file("datasets/mnist_train_images.bin", IMAGE_SIZE, &image_count);
    uint8_t *labels = load_binary_file("datasets/mnist_train_labels.bin", 1, &label_count);
    printf("Loaded %ld training images.\n\n", image_count);

    BatchLoader loader = {image_pixels, labels, image_count, malloc(image_count * sizeof(long)), 0};
    for (long i = 0; i < image_count; i++) {
        loader.visiting_order[i] = i;
    }

    /* The whole-dataset label mix, the reference for the fairness check. */
    long dataset_label_counts[10] = {0};
    for (long i = 0; i < image_count; i++) {
        dataset_label_counts[labels[i]]++;
    }

    static double batch_pixels[BATCH_SIZE * IMAGE_SIZE];
    static uint8_t batch_labels[BATCH_SIZE];
    uint64_t random_state = 42;

    /* Property 1: one epoch touches every example exactly once. */
    long *visit_counts = calloc(image_count, sizeof(long));
    begin_epoch(&loader, &random_state);
    long batches_this_epoch = 0;
    long first_batch_label_counts[10] = {0};
    int batch_count;
    while ((batch_count = next_batch(&loader, batch_pixels, batch_labels)) > 0) {
        if (batches_this_epoch == 0) {
            for (int i = 0; i < batch_count; i++) {
                first_batch_label_counts[batch_labels[i]]++;
            }
        }
        /* next_position - batch_count is where this batch started; count the
         * visits through the visiting order to audit the loader. */
        for (int i = 0; i < batch_count; i++) {
            visit_counts[loader.visiting_order[loader.next_position - batch_count + i]]++;
        }
        batches_this_epoch++;
    }
    long visited_exactly_once = 0;
    for (long i = 0; i < image_count; i++) {
        visited_exactly_once += visit_counts[i] == 1;
    }
    printf("Property 1 - complete coverage: %ld batches delivered %ld examples;\n",
           batches_this_epoch, loader.next_position);
    printf("  examples visited exactly once: %ld of %ld %s\n\n",
           visited_exactly_once, image_count,
           visited_exactly_once == image_count ? "(PASS)" : "(FAIL)");

    /* Property 2: any single batch's label mix resembles the dataset's. */
    printf("Property 2 - shuffled batches are fair samples (label percentages):\n");
    printf("  digit:        0     1     2     3     4     5     6     7     8     9\n");
    printf("  dataset:  ");
    for (int digit = 0; digit < 10; digit++) {
        printf("%5.1f ", 100.0 * dataset_label_counts[digit] / image_count);
    }
    printf("\n  batch 1:  ");
    for (int digit = 0; digit < 10; digit++) {
        printf("%5.1f ", 100.0 * first_batch_label_counts[digit] / BATCH_SIZE);
    }
    printf("\n  A 100-image random batch tracks the 60,000-image mix within a few points -\n");
    printf("  which is why a mini-batch gradient points roughly like the full gradient (Chapter 9).\n\n");

    /* And the transform step: prove the normalization happened. */
    double minimum_pixel = 1.0, maximum_pixel = 0.0;
    for (int i = 0; i < BATCH_SIZE * IMAGE_SIZE; i++) {
        if (batch_pixels[i] < minimum_pixel) minimum_pixel = batch_pixels[i];
        if (batch_pixels[i] > maximum_pixel) maximum_pixel = batch_pixels[i];
    }
    printf("Transform check: delivered pixels lie in [%.2f, %.2f] (bytes 0-255 scaled to 0-1).\n",
           minimum_pixel, maximum_pixel);

    free(image_pixels);
    free(labels);
    free(loader.visiting_order);
    free(visit_counts);
    return 0;
}
