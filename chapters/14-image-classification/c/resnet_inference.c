/*
 * Chapter 14 - the trained ResNet, running in pure C.
 *
 * Reads the folded weights written by export_for_c.py (batch norms already
 * merged into the convolutions - see that script for the math) and runs the
 * exact SmallResNet architecture on 1,000 CIFAR-10 test images. The accuracy
 * matches the Python evaluation to within float32 rounding, because it is
 * the same arithmetic: convolutions, ReLUs, residual adds, one average, one
 * linear layer.
 *
 * Before the first run (from the repository root):
 *     .venv/bin/python chapters/14-image-classification/python/train_cifar10_resnet.py
 *     .venv/bin/python chapters/14-image-classification/python/export_for_c.py
 *
 * Build and run from the repository root:
 *     make -C chapters/14-image-classification/c
 *     ./chapters/14-image-classification/c/build/resnet_inference
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IMAGE_SIZE 32
#define CLASS_COUNT 10

/* The largest feature map is 16 channels x 32 x 32; buffers sized for the
 * largest channel count (64) at every resolution stay comfortably small. */
#define MAX_MAP_FLOATS (64 * 32 * 32)

static const float channel_means[3] = {0.4914f, 0.4822f, 0.4465f};
static const float channel_stds[3] = {0.2470f, 0.2435f, 0.2616f};

static const char *class_names[CLASS_COUNT] = {
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck",
};

/* One folded convolution's parameters, pointing into the loaded weights blob. */
typedef struct {
    const float *weights;   /* out_channels * in_channels * kernel * kernel */
    const float *biases;    /* out_channels */
    int input_channels;
    int output_channels;
    int kernel_size;        /* 3 or 1 */
    int stride;
} FoldedConvolution;

/*
 * Convolution + bias, optional ReLU - the single workhorse of the network.
 * Padding is kernel_size/2 ("same" for odd kernels), matching PyTorch's
 * padding=1 for 3x3 and padding=0 for 1x1.
 *
 * layer:        the folded convolution to apply
 * input_map:    input_channels * input_size * input_size values
 * input_size:   spatial side length of the input map
 * output_map:   receives output_channels * output_size * output_size values,
 *               where output_size = input_size / stride
 * apply_relu:   1 to clamp negatives to zero after the bias
 */
static void run_convolution(const FoldedConvolution *layer, const float *input_map, int input_size,
                            float *output_map, int apply_relu) {
    int padding = layer->kernel_size / 2;
    int output_size = input_size / layer->stride;

    for (int out_channel = 0; out_channel < layer->output_channels; out_channel++) {
        for (int out_row = 0; out_row < output_size; out_row++) {
            for (int out_column = 0; out_column < output_size; out_column++) {
                float accumulated_sum = layer->biases[out_channel];
                for (int in_channel = 0; in_channel < layer->input_channels; in_channel++) {
                    for (int kernel_row = 0; kernel_row < layer->kernel_size; kernel_row++) {
                        int in_row = out_row * layer->stride + kernel_row - padding;
                        if (in_row < 0 || in_row >= input_size) {
                            continue;  /* zero padding: contributions are zero, skip them */
                        }
                        for (int kernel_column = 0; kernel_column < layer->kernel_size; kernel_column++) {
                            int in_column = out_column * layer->stride + kernel_column - padding;
                            if (in_column < 0 || in_column >= input_size) {
                                continue;
                            }
                            float input_value = input_map[(in_channel * input_size + in_row) * input_size + in_column];
                            float weight = layer->weights[((out_channel * layer->input_channels + in_channel)
                                                          * layer->kernel_size + kernel_row)
                                                          * layer->kernel_size + kernel_column];
                            accumulated_sum += input_value * weight;
                        }
                    }
                }
                if (apply_relu && accumulated_sum < 0.0f) {
                    accumulated_sum = 0.0f;
                }
                output_map[(out_channel * output_size + out_row) * output_size + out_column] = accumulated_sum;
            }
        }
    }
}

/*
 * A residual block: out = relu( conv2(relu(conv1(x))) + shortcut(x) ).
 *
 * conv1, conv2:  the block's two folded 3x3 convolutions
 * shortcut:      the folded 1x1 shortcut convolution, or NULL when the
 *                shortcut is the identity
 * feature_map:   in/out: the current map (input_channels * size * size in,
 *                output_channels * out_size * out_size out)
 * input_size:    spatial side length going in
 *
 * Returns the output spatial size. Uses static scratch buffers - fine here
 * because inference runs one image at a time on one thread.
 */
static int run_residual_block(const FoldedConvolution *conv1, const FoldedConvolution *conv2,
                              const FoldedConvolution *shortcut, float *feature_map, int input_size) {
    static float main_path[MAX_MAP_FLOATS];
    static float main_path_2[MAX_MAP_FLOATS];
    static float shortcut_path[MAX_MAP_FLOATS];

    int output_size = input_size / conv1->stride;
    int output_values = conv2->output_channels * output_size * output_size;

    run_convolution(conv1, feature_map, input_size, main_path, 1);
    run_convolution(conv2, main_path, output_size, main_path_2, 0);

    if (shortcut != NULL) {
        run_convolution(shortcut, feature_map, input_size, shortcut_path, 0);
    } else {
        memcpy(shortcut_path, feature_map, output_values * sizeof(float));
    }

    /* The residual add, then the block's final ReLU. Addition is also why
     * gradients flowed so freely during training (Chapter 8's add rule). */
    for (int i = 0; i < output_values; i++) {
        float summed = main_path_2[i] + shortcut_path[i];
        feature_map[i] = summed > 0.0f ? summed : 0.0f;
    }
    return output_size;
}

/*
 * Load a whole binary file (Chapter 9's helper).
 */
static uint8_t *load_binary_file(const char *file_path, long *size_out) {
    FILE *file = fopen(file_path, "rb");
    if (file == NULL) {
        fprintf(stderr, "Cannot open %s - run the training and export scripts first (see the top of this file).\n",
                file_path);
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
    if (size_out != NULL) {
        *size_out = file_size;
    }
    return buffer;
}

int main(void) {
    long weights_size;
    uint8_t *weights_blob = load_binary_file("datasets/cifar10_resnet_weights.bin", &weights_size);
    const int32_t *header = (const int32_t *)weights_blob;
    if (header[0] != 0x524E3134) {
        fprintf(stderr, "Wrong weights file (bad magic number).\n");
        return 1;
    }
    printf("Loaded weights: %ld bytes, %d folded convolutions + linear head.\n", weights_size, header[1]);

    /* Walk the blob in the exact order export_for_c.py wrote it, wiring up
     * each layer's pointers. The architecture is fixed, so the geometry
     * table below IS the network definition. */
    struct { int in_ch, out_ch, kernel, stride; } geometry[] = {
        {3, 16, 3, 1},                                   /* stem */
        {16, 16, 3, 1}, {16, 16, 3, 1},                  /* stage1 block1 */
        {16, 16, 3, 1}, {16, 16, 3, 1},                  /* stage1 block2 */
        {16, 32, 3, 2}, {32, 32, 3, 1}, {16, 32, 1, 2},  /* stage2 block1 + shortcut */
        {32, 32, 3, 1}, {32, 32, 3, 1},                  /* stage2 block2 */
        {32, 64, 3, 2}, {64, 64, 3, 1}, {32, 64, 1, 2},  /* stage3 block1 + shortcut */
        {64, 64, 3, 1}, {64, 64, 3, 1},                  /* stage3 block2 */
    };
    const int convolution_count = 15;
    FoldedConvolution layers[15];
    const float *cursor = (const float *)(weights_blob + 8);
    for (int layer_index = 0; layer_index < convolution_count; layer_index++) {
        layers[layer_index].input_channels = geometry[layer_index].in_ch;
        layers[layer_index].output_channels = geometry[layer_index].out_ch;
        layers[layer_index].kernel_size = geometry[layer_index].kernel;
        layers[layer_index].stride = geometry[layer_index].stride;
        layers[layer_index].weights = cursor;
        cursor += (long)geometry[layer_index].out_ch * geometry[layer_index].in_ch
                * geometry[layer_index].kernel * geometry[layer_index].kernel;
        layers[layer_index].biases = cursor;
        cursor += geometry[layer_index].out_ch;
    }
    const float *linear_weights = cursor;             /* 10 x 64 */
    const float *linear_biases = cursor + 10 * 64;

    uint8_t *image_bytes = load_binary_file("datasets/cifar10_test_images.bin", NULL);
    uint8_t *labels = load_binary_file("datasets/cifar10_test_labels.bin", NULL);
    const int image_count = 1000;

    static float feature_map[MAX_MAP_FLOATS];
    static float stem_output[MAX_MAP_FLOATS];
    int correct_count = 0;

    printf("Classifying %d test images...\n\n", image_count);
    for (int image_index = 0; image_index < image_count; image_index++) {
        /* Normalize exactly like the Python pipeline: byte/255, then
         * per-channel (value - mean) / std. */
        const uint8_t *pixels = image_bytes + (long)image_index * 3 * IMAGE_SIZE * IMAGE_SIZE;
        for (int channel = 0; channel < 3; channel++) {
            for (int i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
                feature_map[channel * IMAGE_SIZE * IMAGE_SIZE + i] =
                    ((float)pixels[channel * IMAGE_SIZE * IMAGE_SIZE + i] / 255.0f - channel_means[channel])
                    / channel_stds[channel];
            }
        }

        run_convolution(&layers[0], feature_map, IMAGE_SIZE, stem_output, 1);
        memcpy(feature_map, stem_output, 16 * IMAGE_SIZE * IMAGE_SIZE * sizeof(float));

        int map_size = IMAGE_SIZE;
        map_size = run_residual_block(&layers[1], &layers[2], NULL, feature_map, map_size);
        map_size = run_residual_block(&layers[3], &layers[4], NULL, feature_map, map_size);
        map_size = run_residual_block(&layers[5], &layers[6], &layers[7], feature_map, map_size);
        map_size = run_residual_block(&layers[8], &layers[9], NULL, feature_map, map_size);
        map_size = run_residual_block(&layers[10], &layers[11], &layers[12], feature_map, map_size);
        map_size = run_residual_block(&layers[13], &layers[14], NULL, feature_map, map_size);

        /* Global average pooling: each channel collapses to its mean. */
        float pooled_features[64];
        int values_per_channel = map_size * map_size;
        for (int channel = 0; channel < 64; channel++) {
            float channel_sum = 0.0f;
            for (int i = 0; i < values_per_channel; i++) {
                channel_sum += feature_map[channel * values_per_channel + i];
            }
            pooled_features[channel] = channel_sum / values_per_channel;
        }

        /* The linear head, then argmax. */
        int predicted_class = 0;
        float best_score = -1e30f;
        for (int class_index = 0; class_index < CLASS_COUNT; class_index++) {
            float score = linear_biases[class_index];
            for (int feature_index = 0; feature_index < 64; feature_index++) {
                score += linear_weights[class_index * 64 + feature_index] * pooled_features[feature_index];
            }
            if (score > best_score) {
                best_score = score;
                predicted_class = class_index;
            }
        }

        correct_count += predicted_class == labels[image_index];
        if (image_index < 5) {
            printf("  image %d: true %-10s predicted %-10s %s\n", image_index,
                   class_names[labels[image_index]], class_names[predicted_class],
                   predicted_class == labels[image_index] ? "(correct)" : "(wrong)");
        }
    }

    printf("\nC inference accuracy: %.2f%% on %d images - compare with the Python evaluation.\n",
           100.0 * correct_count / image_count, image_count);
    printf("Same folded arithmetic, same answers. Batch norm vanished into the weights.\n");

    free(weights_blob);
    free(image_bytes);
    free(labels);
    return 0;
}
