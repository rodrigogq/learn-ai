/*
 * Chapter 16 - mask post-processing in pure C: from pixels to objects.
 *
 * A segmentation network outputs a class per pixel - but applications want
 * OBJECTS: "two circles, here and here, this big". The bridge is classic
 * computer science, no learning involved: connected-component labeling
 * (flood fill) groups touching same-class pixels into instances, and from
 * each instance the area, centroid, and bounding box fall out by counting.
 *
 * This program builds a small demo mask (as if a U-Net had produced it),
 * finds its components, and prints their statistics - plus one practical
 * cleanup: dropping tiny components as noise.
 *
 * Build and run from the repository root:
 *     make -C chapters/16-segmentation/c
 *     ./chapters/16-segmentation/c/build/mask_postprocessing
 */

#include <stdio.h>

#define MASK_SIZE 24
#define MAX_COMPONENTS 32

/* Class ids match the Python chapter: 0 background, 1 circle, 2 rectangle. */
static int mask[MASK_SIZE][MASK_SIZE];
static int component_labels[MASK_SIZE][MASK_SIZE];

typedef struct {
    int class_id;
    int pixel_count;
    int row_sum, column_sum;                    /* for the centroid */
    int min_row, min_column, max_row, max_column;  /* the bounding box */
} ComponentStatistics;

/*
 * Iterative flood fill from a seed pixel: label every reachable pixel of the
 * same class with the same component id. A stack of pixel coordinates
 * replaces recursion so large components cannot overflow the call stack.
 *
 * seed_row, seed_column: where the fill starts
 * component_id:          the label to paint
 * statistics:            accumulated per-component numbers, updated here
 */
static void flood_fill(int seed_row, int seed_column, int component_id,
                       ComponentStatistics *statistics) {
    static int stack_rows[MASK_SIZE * MASK_SIZE];
    static int stack_columns[MASK_SIZE * MASK_SIZE];
    int stack_top = 0;
    int class_id = mask[seed_row][seed_column];

    stack_rows[stack_top] = seed_row;
    stack_columns[stack_top] = seed_column;
    stack_top++;
    component_labels[seed_row][seed_column] = component_id;

    while (stack_top > 0) {
        stack_top--;
        int row = stack_rows[stack_top];
        int column = stack_columns[stack_top];

        statistics->pixel_count++;
        statistics->row_sum += row;
        statistics->column_sum += column;
        if (row < statistics->min_row) statistics->min_row = row;
        if (row > statistics->max_row) statistics->max_row = row;
        if (column < statistics->min_column) statistics->min_column = column;
        if (column > statistics->max_column) statistics->max_column = column;

        /* 4-connectivity: up, down, left, right. Diagonal-touching pixels
         * count as separate objects - the usual convention for masks. */
        const int row_offsets[4] = {-1, 1, 0, 0};
        const int column_offsets[4] = {0, 0, -1, 1};
        for (int direction = 0; direction < 4; direction++) {
            int next_row = row + row_offsets[direction];
            int next_column = column + column_offsets[direction];
            if (next_row >= 0 && next_row < MASK_SIZE && next_column >= 0 && next_column < MASK_SIZE
                && component_labels[next_row][next_column] == 0
                && mask[next_row][next_column] == class_id) {
                component_labels[next_row][next_column] = component_id;
                stack_rows[stack_top] = next_row;
                stack_columns[stack_top] = next_column;
                stack_top++;
            }
        }
    }
    statistics->class_id = class_id;
}

/*
 * Paint a filled circle or square into the demo mask - standing in for what
 * the U-Net's argmax would produce.
 */
static void paint_shape(int class_id, int center_row, int center_column, int half_size) {
    for (int row = 0; row < MASK_SIZE; row++) {
        for (int column = 0; column < MASK_SIZE; column++) {
            int row_distance = row - center_row;
            int column_distance = column - center_column;
            int inside;
            if (class_id == 1) {
                inside = row_distance * row_distance + column_distance * column_distance
                       <= half_size * half_size;
            } else {
                inside = row_distance >= -half_size && row_distance <= half_size
                      && column_distance >= -half_size && column_distance <= half_size;
            }
            if (inside) {
                mask[row][column] = class_id;
            }
        }
    }
}

int main(void) {
    /* A mask a U-Net might output: two circles, one rectangle, plus a
     * 2-pixel speck of noise (misclassified pixels happen). */
    paint_shape(1, 6, 6, 4);
    paint_shape(1, 16, 18, 3);
    paint_shape(2, 16, 6, 4);
    mask[2][20] = 2;
    mask[2][21] = 2;

    printf("The input mask ('.'=background 'o'=circle '#'=rectangle):\n");
    const char symbols[3] = {'.', 'o', '#'};
    for (int row = 0; row < MASK_SIZE; row++) {
        printf("   ");
        for (int column = 0; column < MASK_SIZE; column++) {
            printf("%c", symbols[mask[row][column]]);
        }
        printf("\n");
    }

    /* Connected-component labeling: scan for unlabeled non-background pixels
     * and flood-fill from each - one fill per object instance. */
    ComponentStatistics statistics[MAX_COMPONENTS];
    int component_count = 0;
    for (int row = 0; row < MASK_SIZE; row++) {
        for (int column = 0; column < MASK_SIZE; column++) {
            if (mask[row][column] != 0 && component_labels[row][column] == 0
                && component_count < MAX_COMPONENTS) {
                statistics[component_count] = (ComponentStatistics){
                    0, 0, 0, 0, MASK_SIZE, MASK_SIZE, -1, -1,
                };
                flood_fill(row, column, component_count + 1, &statistics[component_count]);
                component_count++;
            }
        }
    }

    const char *class_names[3] = {"background", "circle", "rectangle"};
    const int minimum_pixels_to_keep = 4;
    printf("\nComponents found: %d\n", component_count);
    for (int i = 0; i < component_count; i++) {
        ComponentStatistics *component = &statistics[i];
        double centroid_row = (double)component->row_sum / component->pixel_count;
        double centroid_column = (double)component->column_sum / component->pixel_count;
        printf("  #%d: %-9s  area %3d px  centroid (%.1f, %.1f)  box rows %d-%d cols %d-%d%s\n",
               i + 1, class_names[component->class_id], component->pixel_count,
               centroid_row, centroid_column,
               component->min_row, component->max_row, component->min_column, component->max_column,
               component->pixel_count < minimum_pixels_to_keep
                   ? "   -> dropped as noise (< 4 px)" : "");
    }

    printf("\nThis is how a per-pixel mask becomes 'two circles and a rectangle, here,\n");
    printf("this big' - counting objects, measuring areas, or driving a robot arm.\n");
    return 0;
}
