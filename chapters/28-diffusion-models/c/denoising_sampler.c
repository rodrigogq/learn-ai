/*
 * Chapter 28 - the diffusion sampling loop in pure C.
 *
 * Generation in diffusion is not one forward pass (like a GAN) but a LOOP:
 * start from pure noise and repeatedly nudge the image toward "less noisy"
 * using the denoiser's predictions. This program implements that reverse loop
 * with a stand-in denoiser (a hand-built one that pulls pixels toward a target
 * digit shape), so the ALGORITHM - the schedule, the step, the added noise -
 * is fully visible without needing to ship trained U-Net weights. A real
 * diffusion sampler is this exact loop with a neural network in place of the
 * stand-in.
 *
 * It prints the image at several points in the reverse process so you watch
 * structure emerge from static.
 *
 * Build and run from the repository root:
 *     make -C chapters/28-diffusion-models/c
 *     ./chapters/28-diffusion-models/c/build/denoising_sampler
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define IMAGE_SIDE 16
#define IMAGE_SIZE (IMAGE_SIDE * IMAGE_SIDE)
#define STEPS 60

static uint64_t random_state = 7;
static double next_uniform(void) {
    random_state = random_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(random_state >> 11) * (1.0 / 9007199254740992.0);
}
/* Box-Muller: one standard-normal sample from two uniforms. */
static float next_normal(void) {
    float u1 = (float)next_uniform();
    float u2 = (float)next_uniform();
    if (u1 < 1e-7f) u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* The target shape the stand-in denoiser "knows" - a plus sign, standing in
 * for what a trained network learned from data. */
static float target_pixel(int row, int column) {
    int center = IMAGE_SIDE / 2;
    int on_vertical = column >= center - 1 && column <= center + 1 && row >= 3 && row < IMAGE_SIDE - 3;
    int on_horizontal = row >= center - 1 && row <= center + 1 && column >= 3 && column < IMAGE_SIDE - 3;
    return (on_vertical || on_horizontal) ? 1.0f : 0.0f;
}

/*
 * The stand-in denoiser: given a noisy image and how noisy it is (noise_level
 * 1.0 = pure noise, 0.0 = clean), PREDICT the noise present. A trained network
 * learns this from data; here we compute it from the known target - noise is
 * whatever is not the target shape - which is exactly what the network
 * approximates.
 *
 * noisy:        current IMAGE_SIZE pixels
 * noise_level:  how much of the image is still noise
 * predicted_noise_out: receives the estimated noise per pixel
 */
static void predict_noise(const float *noisy, float noise_level, float *predicted_noise_out) {
    for (int row = 0; row < IMAGE_SIDE; row++) {
        for (int column = 0; column < IMAGE_SIDE; column++) {
            int i = row * IMAGE_SIDE + column;
            /* estimated clean image is a blend toward the target that grows as
             * we get less noisy; the predicted noise is what remains. */
            float estimated_clean = (1.0f - noise_level) * target_pixel(row, column) + noise_level * noisy[i];
            predicted_noise_out[i] = noisy[i] - estimated_clean;
        }
    }
}

static void print_image(const float *pixels, const char *label) {
    const char characters[] = " .:-=+*#%@";
    printf("%s\n", label);
    for (int row = 0; row < IMAGE_SIDE; row++) {
        printf("   ");
        for (int column = 0; column < IMAGE_SIDE; column++) {
            float value = pixels[row * IMAGE_SIDE + column];
            if (value < 0.0f) value = 0.0f;
            if (value > 1.0f) value = 1.0f;
            int level = (int)(value * 9);
            putchar(characters[level]);
            putchar(characters[level]);
        }
        putchar('\n');
    }
    printf("\n");
}

int main(void) {
    printf("Diffusion sampling: start from pure noise, denoise step by step.\n");
    printf("(A stand-in denoiser pulls toward a '+' shape; a trained U-Net pulls\n");
    printf(" toward whatever it learned - Chapter 28's Python version, toward digits.)\n\n");

    float image[IMAGE_SIZE];
    float predicted_noise[IMAGE_SIZE];
    for (int i = 0; i < IMAGE_SIZE; i++) image[i] = next_normal();
    print_image(image, "step 60/60: pure noise");

    for (int step = STEPS; step > 0; step--) {
        float noise_level = (float)step / STEPS;
        predict_noise(image, noise_level, predicted_noise);

        /* The reverse step: remove a fraction of the predicted noise, then
         * (except at the very end) add back a little fresh noise. Removing all
         * the noise at once gives blurry averages; the gradual walk with
         * re-injected noise is what yields sharp, varied samples. */
        float removal_fraction = 1.0f / step;
        for (int i = 0; i < IMAGE_SIZE; i++) {
            image[i] -= removal_fraction * predicted_noise[i];
            if (step > 1) {
                image[i] += 0.08f * noise_level * next_normal();
            }
        }

        if (step == 40 || step == 20 || step == 5) {
            char label[64];
            snprintf(label, sizeof(label), "step %d/60: structure emerging", step);
            print_image(image, label);
        }
    }
    print_image(image, "step 0/60: the sample");

    printf("Same idea as the Python U-Net, and as Stable Diffusion: generation is a\n");
    printf("LOOP of small denoising steps, not a single shot. Slow, but the most stable\n");
    printf("and highest-quality route to images we have.\n");
    return 0;
}
