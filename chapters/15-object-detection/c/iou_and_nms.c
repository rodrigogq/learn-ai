/*
 * Chapter 15 - IoU and non-maximum suppression from scratch.
 *
 * These two small algorithms are the universal plumbing of object detection:
 * every detector, from this chapter's toy to production YOLO models, measures
 * box overlap with IoU and removes duplicate detections with NMS. Both fit on
 * one screen, and this program reproduces the chapter's worked examples.
 *
 * Build and run from the repository root:
 *     make -C chapters/15-object-detection/c
 *     ./chapters/15-object-detection/c/build/iou_and_nms
 */

#include <stdio.h>

typedef struct {
    double x_min, y_min, x_max, y_max;
} Box;

typedef struct {
    Box box;
    double confidence_score;
    int class_id;
    int suppressed;   /* set by NMS when a stronger overlapping box wins */
} Detection;

/*
 * Intersection over union: the standard measure of box agreement.
 *
 * first, second: the two boxes, as (x_min, y_min, x_max, y_max).
 *
 * Returns a number in [0, 1]: 0 for disjoint boxes, 1 for identical ones.
 * The intersection rectangle's corners are the max of the mins and the min
 * of the maxes; a negative width or height means no overlap at all.
 */
static double compute_iou(Box first, Box second) {
    double intersection_left = first.x_min > second.x_min ? first.x_min : second.x_min;
    double intersection_top = first.y_min > second.y_min ? first.y_min : second.y_min;
    double intersection_right = first.x_max < second.x_max ? first.x_max : second.x_max;
    double intersection_bottom = first.y_max < second.y_max ? first.y_max : second.y_max;

    double intersection_width = intersection_right - intersection_left;
    double intersection_height = intersection_bottom - intersection_top;
    if (intersection_width <= 0.0 || intersection_height <= 0.0) {
        return 0.0;
    }
    double intersection_area = intersection_width * intersection_height;
    double first_area = (first.x_max - first.x_min) * (first.y_max - first.y_min);
    double second_area = (second.x_max - second.x_min) * (second.y_max - second.y_min);
    return intersection_area / (first_area + second_area - intersection_area);
}

/*
 * Non-maximum suppression, the greedy version every detector uses:
 * repeatedly take the highest-confidence unsuppressed detection, keep it,
 * and suppress every unkept detection that overlaps it above the threshold.
 *
 * detections:      the candidate list (modified in place: .suppressed is set)
 * detection_count: how many candidates there are
 * iou_threshold:   overlap above which the weaker detection is a duplicate
 */
static void run_non_maximum_suppression(Detection *detections, int detection_count,
                                        double iou_threshold) {
    for (;;) {
        /* Find the best-scoring detection not yet kept or suppressed. */
        int best_index = -1;
        for (int i = 0; i < detection_count; i++) {
            if (detections[i].suppressed == 0
                && (best_index == -1 || detections[i].confidence_score > detections[best_index].confidence_score)) {
                best_index = i;
            }
        }
        if (best_index == -1) {
            break;
        }
        detections[best_index].suppressed = -1;  /* -1 marks "kept" */

        for (int i = 0; i < detection_count; i++) {
            if (detections[i].suppressed == 0
                && compute_iou(detections[best_index].box, detections[i].box) > iou_threshold) {
                detections[i].suppressed = 1;
            }
        }
    }
}

int main(void) {
    printf("1. IoU worked example (the chapter's numbers)\n");
    Box box_a = {20, 20, 60, 60};
    Box box_b = {40, 30, 80, 70};
    printf("   box A = (20,20)-(60,60), box B = (40,30)-(80,70)\n");
    printf("   intersection: 20 x 30 = 600; union: 1600 + 1600 - 600 = 2600\n");
    printf("   IoU = 600 / 2600 = %.3f\n", compute_iou(box_a, box_b));

    Box far_away = {100, 100, 120, 120};
    printf("   IoU of A with a distant box: %.3f (no overlap)\n", compute_iou(box_a, far_away));
    printf("   IoU of A with itself:        %.3f (perfect)\n\n", compute_iou(box_a, box_a));

    printf("2. NMS on five candidate detections (threshold 0.5)\n");
    Detection detections[] = {
        /* Three near-duplicates on the same digit: neighboring grid cells
         * all fired. Only the strongest should survive. */
        {{18, 20, 46, 48}, 0.92, 7, 0},
        {{20, 22, 48, 50}, 0.75, 7, 0},
        {{16, 18, 44, 46}, 0.61, 7, 0},
        /* A separate object elsewhere in the image. */
        {{50, 50, 78, 78}, 0.83, 3, 0},
        /* A weak stray far from everything - kept by NMS (NMS only removes
         * DUPLICATES; low-confidence strays are the threshold's job). */
        {{5, 50, 20, 65}, 0.55, 1, 0},
    };
    const int detection_count = 5;

    printf("   before: %d candidates\n", detection_count);
    run_non_maximum_suppression(detections, detection_count, 0.5);
    printf("   after:\n");
    for (int i = 0; i < detection_count; i++) {
        printf("     score %.2f class %d box (%.0f,%.0f)-(%.0f,%.0f): %s\n",
               detections[i].confidence_score, detections[i].class_id,
               detections[i].box.x_min, detections[i].box.y_min,
               detections[i].box.x_max, detections[i].box.y_max,
               detections[i].suppressed == -1 ? "KEPT" : "suppressed (duplicate of a stronger box)");
    }
    printf("\n   Three overlapping candidates for the digit 7 collapsed to one - the reason\n");
    printf("   every detector runs NMS: neighboring cells legitimately fire on the same object.\n");
    return 0;
}
