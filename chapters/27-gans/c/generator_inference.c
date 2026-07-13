/*
 * Chapter 27 - a GAN generator's forward pass in pure C: noise in, image out.
 *
 * Once a GAN is trained, generating is just the generator's forward pass -
 * random numbers through a stack of (transposed) convolutions to pixels, no
 * discriminator, no randomness beyond the input noise. This program shows the
 * mechanism with a compact hand-built generator (self-contained, no data file
 * needed): a noise vector drives a small upsampling pipeline that paints a
 * ring whose size and brightness the noise controls, so you can SEE the noise
 * steering the image. A trained DCGAN generator is this exact shape - noise ->
 * project -> upsample -> pixels - with learned weights.
 *
 * Build and run from the repository root:
 *     make -C chapters/27-gans/c
 *     ./chapters/27-gans/c/build/generator_inference
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define NOISE_SIZE 4
#define IMAGE_SIDE 20
#define IMAGE_SIZE (IMAGE_SIDE * IMAGE_SIDE)

/* Chapter 9's deterministic generator, so the "random" seeds are reproducible. */
static double pseudo_random_uniform(uint64_t *state) {
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(*state >> 11) * (1.0 / 9007199254740992.0);
}

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

/*
 * Generate one image from a noise vector.
 *
 * noise:       NOISE_SIZE numbers in roughly -1..1 (the generator's seed)
 * pixels_out:  IMAGE_SIZE numbers in 0..1
 *
 * A real DCGAN generator projects the noise to a small feature map and grows
 * it with transposed convolutions. We compress that to its essence: the noise
 * sets a few interpretable knobs (ring radius, thickness, brightness, tilt),
 * and the output is painted from them - the same "noise controls image"
 * relationship a trained generator learns, made legible.
 */
static void generate(const float *noise, float *pixels_out) {
    float center = (IMAGE_SIDE - 1) / 2.0f;
    float ring_radius = 4.0f + 3.0f * noise[0];      /* noise[0]: how big */
    float ring_thickness = 1.2f + 0.8f * noise[1];   /* noise[1]: how fat */
    float brightness = 0.7f + 0.3f * noise[2];       /* noise[2]: how bright */
    float tilt = noise[3];                           /* noise[3]: squash into an oval */

    for (int row = 0; row < IMAGE_SIDE; row++) {
        for (int column = 0; column < IMAGE_SIDE; column++) {
            float dy = (row - center) * (1.0f + 0.3f * tilt);
            float dx = (column - center) * (1.0f - 0.3f * tilt);
            float distance = sqrtf(dx * dx + dy * dy);
            /* Bright where the distance is near the ring radius (a soft shell). */
            float closeness = 1.0f - fabsf(distance - ring_radius) / ring_thickness;
            pixels_out[row * IMAGE_SIDE + column] = brightness * sigmoid(6.0f * closeness - 2.0f);
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
        }
        putchar('\n');
    }
}

int main(void) {
    printf("A GAN generator: random noise in, an image out (pure C, no discriminator).\n");
    printf("Each different noise vector yields a different image - as with a trained\n");
    printf("DCGAN, whose learned weights map noise to handwritten digits.\n\n");

    uint64_t random_state = 42;
    float noise[NOISE_SIZE];
    float pixels[IMAGE_SIZE];

    for (int sample = 0; sample < 3; sample++) {
        printf("noise seed: [");
        for (int i = 0; i < NOISE_SIZE; i++) {
            noise[i] = (float)(pseudo_random_uniform(&random_state) * 2.0 - 1.0);
            printf("%+.2f%s", noise[i], i + 1 < NOISE_SIZE ? " " : "");
        }
        printf("]\n");
        generate(noise, pixels);
        print_image(pixels);
        printf("\n");
    }

    printf("Three seeds, three different images - the generator is a deterministic\n");
    printf("function of its noise. Train it adversarially and those images become\n");
    printf("indistinguishable from real data. That is a GAN.\n");
    return 0;
}
