/*
 * Chapter 29 - classifier-free guidance and frame consistency, in pure C.
 *
 * Two mechanisms of conditional generation, made concrete with stand-in
 * denoisers (self-contained, no trained weights needed):
 *
 *   1. CLASSIFIER-FREE GUIDANCE - the "how hard should it obey the prompt"
 *      knob. The final prediction is
 *          uncond + scale * (cond - uncond)
 *      Turning 'scale' up amplifies whatever the condition asked for. We show
 *      the same seed generated at several guidance scales, so you see weak
 *      guidance ignore the request and strong guidance enforce it.
 *
 *   2. FRAME CONSISTENCY - why sharing a noise seed across frames keeps a
 *      video from flickering. Two frames from the same seed with a slightly
 *      shifted condition stay similar; two frames from DIFFERENT seeds look
 *      unrelated even at the same condition.
 *
 * Build and run from the repository root:
 *     make -C chapters/29-text-to-image-and-video/c
 *     ./chapters/29-text-to-image-and-video/c/build/guidance_and_frames
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define IMAGE_SIDE 18
#define IMAGE_SIZE (IMAGE_SIDE * IMAGE_SIDE)

static uint64_t random_state;
static void seed_rng(uint64_t s) { random_state = s; }
static double next_uniform(void) {
    random_state = random_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(random_state >> 11) * (1.0 / 9007199254740992.0);
}

/*
 * A stand-in "conditioned prediction": the target image for a given condition.
 * condition in [0,1] slides a bright bar from the top of the frame to the
 * bottom, standing in for "which digit / which prompt". A real model's
 * conditioned prediction is a learned function; the shape does not matter,
 * only that it depends on the condition.
 */
static float conditioned_target(int row, int column, float condition) {
    int bar_row = (int)(condition * (IMAGE_SIDE - 4)) + 2;
    int on_bar = row >= bar_row - 1 && row <= bar_row + 1 && column >= 3 && column < IMAGE_SIDE - 3;
    return on_bar ? 1.0f : 0.0f;
}

/* The unconditioned prediction ignores the condition - a vague blob in the
 * middle, "an image, but no particular one". */
static float unconditioned_target(int row, int column) {
    float dr = row - IMAGE_SIDE / 2.0f, dc = column - IMAGE_SIDE / 2.0f;
    return expf(-(dr * dr + dc * dc) / 40.0f) * 0.5f;
}

/*
 * Generate an image from a seed at a given condition and guidance scale, by
 * running a short denoising loop whose prediction blends the conditioned and
 * unconditioned targets exactly as classifier-free guidance prescribes.
 */
static void generate(uint64_t seed, float condition, float guidance_scale, float *pixels_out) {
    seed_rng(seed);
    for (int i = 0; i < IMAGE_SIZE; i++) {
        pixels_out[i] = (float)(next_uniform() * 2.0 - 1.0);   /* start from noise */
    }
    const int steps = 40;
    for (int step = steps; step > 0; step--) {
        float removal = 1.0f / step;
        for (int row = 0; row < IMAGE_SIDE; row++) {
            for (int column = 0; column < IMAGE_SIDE; column++) {
                int i = row * IMAGE_SIDE + column;
                float cond = conditioned_target(row, column, condition);
                float uncond = unconditioned_target(row, column);
                /* the guidance formula: push away from unconditioned */
                float guided = uncond + guidance_scale * (cond - uncond);
                pixels_out[i] += removal * (guided - pixels_out[i]);
            }
        }
    }
}

static void print_row(float images[][IMAGE_SIZE], const char *labels[], int count) {
    const char characters[] = " .:-=+*#%@";
    for (int i = 0; i < count; i++) printf("  %-*s", IMAGE_SIDE + 3, labels[i]);
    printf("\n");
    for (int row = 0; row < IMAGE_SIDE; row++) {
        for (int i = 0; i < count; i++) {
            printf("  ");
            for (int column = 0; column < IMAGE_SIDE; column++) {
                float v = images[i][row * IMAGE_SIDE + column];
                if (v < 0) v = 0; if (v > 1) v = 1;
                putchar(characters[(int)(v * 9)]);
            }
            printf(" ");
        }
        printf("\n");
    }
    printf("\n");
}

int main(void) {
    static float images[4][IMAGE_SIZE];

    printf("1. Classifier-free guidance: the same seed and condition, rising guidance scale\n\n");
    const char *guidance_labels[] = {"scale 0 (ignore)", "scale 1", "scale 3", "scale 6 (strong)"};
    float scales[] = {0.0f, 1.0f, 3.0f, 6.0f};
    for (int i = 0; i < 4; i++) {
        generate(2024, 0.7f, scales[i], images[i]);
    }
    print_row(images, guidance_labels, 4);
    printf("  Left: guidance 0 ignores the condition (a vague blob). Right: strong guidance\n");
    printf("  enforces it (the requested bar, sharp). This is a text model's 'guidance scale'.\n\n");

    printf("2. Frame consistency: shared seed vs different seeds, condition sliding 0.3 -> 0.6\n\n");
    static float shared[4][IMAGE_SIZE];
    static float different[4][IMAGE_SIZE];
    for (int f = 0; f < 4; f++) {
        float condition = 0.3f + 0.1f * f;
        generate(555, condition, 3.0f, shared[f]);          /* same seed every frame */
        generate(555 + f * 99991, condition, 3.0f, different[f]);  /* a new seed per frame */
    }
    const char *frame_labels[] = {"frame 1", "frame 2", "frame 3", "frame 4"};
    printf("  shared seed (smooth 'video'):\n");
    print_row(shared, frame_labels, 4);
    printf("  different seeds (flickery - unusable as video):\n");
    print_row(different, frame_labels, 4);
    printf("  Sharing the seed is why the bar glides smoothly; different seeds jump around.\n");
    printf("  Real video models add a temporal network on top, but this shared-latent idea\n");
    printf("  is the foundation of frame-to-frame consistency.\n");
    return 0;
}
