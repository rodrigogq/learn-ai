/*
 * Chapter 26 - a generative decoder in pure C: turn 2 numbers into a digit.
 *
 * The generative half of a VAE is just its decoder - a small MLP from a
 * latent point to 784 pixels. Once trained, GENERATING an image is one
 * forward pass and no randomness beyond choosing the latent point. This
 * program hard-codes a tiny trained decoder's weights (exported separately)
 * would normally be loaded; here we demonstrate the mechanism with a compact
 * hand-built decoder so the file is self-contained and needs no data.
 *
 * The point it makes: image generation, stripped of the framework, is a
 * weighted-sum pipeline (Chapter 0) ending in pixels. The C engine of
 * Chapter 25 does the same for text; this does it for images.
 *
 * Build and run from the repository root:
 *     make -C chapters/26-autoencoders-and-vaes/c
 *     ./chapters/26-autoencoders-and-vaes/c/build/decoder_inference
 */

#include <math.h>
#include <stdio.h>

#define LATENT_SIZE 2
#define HIDDEN_SIZE 16
#define IMAGE_SIDE 16
#define IMAGE_SIZE (IMAGE_SIDE * IMAGE_SIDE)

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float relu(float x) { return x > 0.0f ? x : 0.0f; }

/*
 * A tiny hand-designed decoder: latent (2) -> hidden (16, ReLU) -> pixels
 * (256, sigmoid). The weights are chosen (not trained) so that the two latent
 * axes control interpretable things - axis 0 draws a bright vertical bar
 * whose position follows the value, axis 1 a horizontal bar - so you can SEE
 * the latent space controlling the image. A trained VAE decoder is this exact
 * shape with learned weights instead.
 *
 * latent:      LATENT_SIZE numbers - the point in latent space
 * pixels_out:  IMAGE_SIZE numbers in 0..1
 */
static void decode(const float *latent, float *pixels_out) {
    /* Hidden layer: each of 16 hidden units responds to a band of positions,
     * driven by the two latent coordinates. */
    float hidden[HIDDEN_SIZE];
    for (int h = 0; h < HIDDEN_SIZE; h++) {
        /* Hidden unit h "prefers" a target coordinate; it lights up when the
         * latent point is near that target along one axis. */
        float target = (h / (float)(HIDDEN_SIZE - 1)) * 2.0f - 1.0f;   /* -1..1 */
        float axis0_match = 1.0f - fabsf(latent[0] - target);
        float axis1_match = 1.0f - fabsf(latent[1] - target);
        hidden[h] = relu(axis0_match) + relu(axis1_match);
    }

    /* Output layer: paint a vertical bar at the column the latent[0] selects
     * and a horizontal bar at the row latent[1] selects, blended through the
     * hidden activations. */
    int bar_column = (int)((latent[0] + 1.0f) * 0.5f * (IMAGE_SIDE - 1));
    int bar_row = (int)((latent[1] + 1.0f) * 0.5f * (IMAGE_SIDE - 1));
    for (int row = 0; row < IMAGE_SIDE; row++) {
        for (int column = 0; column < IMAGE_SIDE; column++) {
            float on_vertical = 1.0f - fabsf(column - bar_column) / 2.0f;
            float on_horizontal = 1.0f - fabsf(row - bar_row) / 2.0f;
            float hidden_influence = hidden[(row + column) % HIDDEN_SIZE] * 0.05f;
            pixels_out[row * IMAGE_SIDE + column] =
                sigmoid(4.0f * (relu(on_vertical) + relu(on_horizontal)) - 3.0f + hidden_influence);
        }
    }
}

static void print_image(const float *pixels) {
    const char characters[] = " .:-=+*#%@";
    for (int row = 0; row < IMAGE_SIDE; row++) {
        printf("   ");
        for (int column = 0; column < IMAGE_SIDE; column++) {
            int level = (int)(pixels[row * IMAGE_SIDE + column] * 9);
            if (level > 9) level = 9;
            putchar(characters[level]);
            putchar(characters[level]);   /* double width so squares look square */
        }
        putchar('\n');
    }
}

int main(void) {
    printf("Generating images by moving through a 2-D latent space.\n");
    printf("(A didactic decoder: axis 0 places a vertical bar, axis 1 a horizontal one.\n");
    printf(" A trained VAE decoder is the same MLP shape with learned weights - Chapter 26's\n");
    printf(" Python version shows real digits emerging the same way.)\n\n");

    float latent_points[][LATENT_SIZE] = {
        {-0.8f, -0.8f}, {0.0f, 0.0f}, {0.8f, 0.8f}, {-0.8f, 0.8f},
    };
    float pixels[IMAGE_SIZE];

    for (int p = 0; p < 4; p++) {
        printf("latent point (%+.1f, %+.1f):\n", latent_points[p][0], latent_points[p][1]);
        decode(latent_points[p], pixels);
        print_image(pixels);
        printf("\n");
    }

    printf("Smoothly changing the two input numbers smoothly moves the image -\n");
    printf("that continuity is exactly what the VAE's KL term buys, and what lets\n");
    printf("you sample new images from nothing but a random point.\n");
    return 0;
}
