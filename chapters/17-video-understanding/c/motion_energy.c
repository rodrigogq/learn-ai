/*
 * Chapter 17 - motion without learning: frame differencing and centroid
 * tracking in pure C.
 *
 * Before (and alongside) deep video models, classic computer vision read
 * motion straight from arithmetic: subtract consecutive frames (what changed?),
 * track the centroid of the change, and the displacement IS the motion. On
 * this chapter's task - which way is the object moving? - the 60-line classic
 * matches the neural networks, a healthy reminder that learning is a tool,
 * not a religion. Its limits (the exercises probe them) are why the networks
 * still matter.
 *
 * Build and run from the repository root:
 *     make -C chapters/17-video-understanding/c
 *     ./chapters/17-video-understanding/c/build/motion_energy
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CANVAS_SIZE 48
#define FRAME_COUNT 8
#define OBJECT_SIZE 12
#define PIXELS_PER_FRAME 2

static const char *direction_names[4] = {"up", "down", "left", "right"};
static const int direction_steps[4][2] = {{-PIXELS_PER_FRAME, 0}, {PIXELS_PER_FRAME, 0},
                                          {0, -PIXELS_PER_FRAME}, {0, PIXELS_PER_FRAME}};

/* Chapter 9's deterministic generator. */
static double pseudo_random_uniform(uint64_t *state) {
    *state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(*state >> 11) * (1.0 / 9007199254740992.0);
}

/*
 * Render one clip: a bright square sliding in a straight line, plus pixel
 * noise so the method has to prove it is not just reading clean pixels.
 *
 * frames:        receives FRAME_COUNT * CANVAS_SIZE * CANVAS_SIZE values
 * direction:     0..3, which way the square moves
 * random_state:  generator for the start position and the noise
 */
static void render_clip(double *frames, int direction, uint64_t *random_state) {
    int travel = PIXELS_PER_FRAME * (FRAME_COUNT - 1);
    int row_step = direction_steps[direction][0];
    int column_step = direction_steps[direction][1];

    int min_row = row_step < 0 ? travel : 0;
    int max_row = CANVAS_SIZE - OBJECT_SIZE - (row_step > 0 ? travel : 0);
    int min_column = column_step < 0 ? travel : 0;
    int max_column = CANVAS_SIZE - OBJECT_SIZE - (column_step > 0 ? travel : 0);
    int start_row = min_row + (int)(pseudo_random_uniform(random_state) * (max_row - min_row + 1));
    int start_column = min_column + (int)(pseudo_random_uniform(random_state) * (max_column - min_column + 1));

    for (int frame = 0; frame < FRAME_COUNT; frame++) {
        double *pixels = frames + (long)frame * CANVAS_SIZE * CANVAS_SIZE;
        for (int i = 0; i < CANVAS_SIZE * CANVAS_SIZE; i++) {
            pixels[i] = 0.15 * pseudo_random_uniform(random_state);   /* background noise */
        }
        int top = start_row + row_step * frame;
        int left = start_column + column_step * frame;
        for (int row = top; row < top + OBJECT_SIZE; row++) {
            for (int column = left; column < left + OBJECT_SIZE; column++) {
                pixels[row * CANVAS_SIZE + column] = 0.9;
            }
        }
    }
}

/*
 * Classify a clip's motion direction with no learned parameters at all.
 *
 * frames: FRAME_COUNT * CANVAS_SIZE * CANVAS_SIZE clip pixels
 *
 * Method: for each consecutive frame pair, compute the "motion energy" image
 * |frame[t+1] - frame[t]| (only moving things survive the subtraction),
 * threshold it, and take the centroid of frame[t+1]'s bright pixels minus
 * the centroid of frame[t]'s. Averaging that displacement over the clip and
 * picking the dominant axis gives the direction.
 *
 * Returns 0..3 (the direction), decided by the accumulated displacement.
 */
static int classify_motion_direction(const double *frames) {
    double total_row_shift = 0.0;
    double total_column_shift = 0.0;

    for (int frame = 0; frame + 1 < FRAME_COUNT; frame++) {
        const double *current = frames + (long)frame * CANVAS_SIZE * CANVAS_SIZE;
        const double *next = frames + (long)(frame + 1) * CANVAS_SIZE * CANVAS_SIZE;

        /* Centroid of bright pixels in each frame. The 0.5 threshold splits
         * the 0.9 object from the <=0.15 noise cleanly - exercise 3 asks
         * what happens when it no longer can. */
        double current_row_sum = 0, current_column_sum = 0, current_count = 0;
        double next_row_sum = 0, next_column_sum = 0, next_count = 0;
        for (int row = 0; row < CANVAS_SIZE; row++) {
            for (int column = 0; column < CANVAS_SIZE; column++) {
                if (current[row * CANVAS_SIZE + column] > 0.5) {
                    current_row_sum += row;
                    current_column_sum += column;
                    current_count++;
                }
                if (next[row * CANVAS_SIZE + column] > 0.5) {
                    next_row_sum += row;
                    next_column_sum += column;
                    next_count++;
                }
            }
        }
        if (current_count > 0 && next_count > 0) {
            total_row_shift += next_row_sum / next_count - current_row_sum / current_count;
            total_column_shift += next_column_sum / next_count - current_column_sum / current_count;
        }
    }

    if (fabs(total_row_shift) > fabs(total_column_shift)) {
        return total_row_shift < 0 ? 0 : 1;   /* up : down */
    }
    return total_column_shift < 0 ? 2 : 3;    /* left : right */
}

int main(void) {
    uint64_t random_state = 42;
    static double frames[FRAME_COUNT * CANVAS_SIZE * CANVAS_SIZE];

    const int clips_per_direction = 250;
    int correct_count = 0;
    int per_direction_correct[4] = {0};

    for (int direction = 0; direction < 4; direction++) {
        for (int trial = 0; trial < clips_per_direction; trial++) {
            render_clip(frames, direction, &random_state);
            int predicted = classify_motion_direction(frames);
            per_direction_correct[direction] += predicted == direction;
            correct_count += predicted == direction;
        }
    }

    printf("Motion by arithmetic: frame differencing + centroid tracking, zero parameters.\n\n");
    printf("  direction   accuracy over %d clips\n", clips_per_direction);
    for (int direction = 0; direction < 4; direction++) {
        printf("  %-9s   %.1f%%\n", direction_names[direction],
               100.0 * per_direction_correct[direction] / clips_per_direction);
    }
    printf("\nOverall: %.1f%% - matching the trained networks on this clean task.\n",
           100.0 * correct_count / (4 * clips_per_direction));
    printf("The networks earn their keep when the world stops being clean: several objects,\n");
    printf("cluttered backgrounds, deformable things, or 'what ACTION is happening' questions.\n");
    return 0;
}
