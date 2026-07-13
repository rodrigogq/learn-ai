"""Chapter 15 - a single-stage object detector, trained on synthetic scenes.

The task: 64x64 images containing 1-3 MNIST digits at random positions; the
model must say WHAT each digit is and WHERE (a bounding box). Architecture is
YOLO-style single-stage: a small conv backbone shrinks the image to an 8x8
grid, and each grid cell predicts [objectness, box, class] for objects whose
center falls inside it. Decoding uses confidence thresholding + non-maximum
suppression (NMS); evaluation counts a detection correct when its box overlaps
the truth with IoU >= 0.5 and the class matches.

Run from the repository root:
    .venv/bin/python chapters/15-object-detection/python/train_digit_detector.py --quick
    .venv/bin/python chapters/15-object-detection/python/train_digit_detector.py
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from common.data import load_mnist_datasets  # noqa: E402
from common.device import select_best_available_device  # noqa: E402

CANVAS_SIZE = 64
GRID_SIZE = 8                       # the backbone shrinks 64 -> 8, so each cell covers 8x8 pixels
CELL_SIZE = CANVAS_SIZE // GRID_SIZE
CLASS_COUNT = 10
# Per cell the head predicts: 1 objectness + 4 box numbers + 10 class scores.
PREDICTION_CHANNELS = 1 + 4 + CLASS_COUNT


def build_scene_batch(digit_images, digit_labels, batch_size, random_generator):
    """Compose synthetic detection scenes from MNIST digits.

    Arguments:
        digit_images: tensor (N, 28, 28) of digit pixels in 0..1.
        digit_labels: tensor (N,) of digit classes.
        batch_size: scenes to build.
        random_generator: torch.Generator for reproducible scenes.

    Returns (scenes, target_maps):
        scenes: (batch, 1, 64, 64) images with 1-3 digits pasted at random.
        target_maps: (batch, PREDICTION_CHANNELS, 8, 8) training targets laid
            out like the model output - objectness 1 at the cell holding each
            digit's center, plus that cell's box parameters and class.
    """
    scenes = torch.zeros(batch_size, 1, CANVAS_SIZE, CANVAS_SIZE)
    target_maps = torch.zeros(batch_size, PREDICTION_CHANNELS, GRID_SIZE, GRID_SIZE)

    for scene_index in range(batch_size):
        digit_count = int(torch.randint(1, 4, (1,), generator=random_generator))
        for _ in range(digit_count):
            source_index = int(torch.randint(len(digit_images), (1,), generator=random_generator))
            digit_image = digit_images[source_index]
            digit_class = int(digit_labels[source_index])

            top = int(torch.randint(0, CANVAS_SIZE - 28, (1,), generator=random_generator))
            left = int(torch.randint(0, CANVAS_SIZE - 28, (1,), generator=random_generator))
            # torch.maximum keeps overlapping digits visible instead of the
            # later paste erasing the earlier one with its black border.
            scenes[scene_index, 0, top:top + 28, left:left + 28] = torch.maximum(
                scenes[scene_index, 0, top:top + 28, left:left + 28], digit_image
            )

            center_x = left + 14.0
            center_y = top + 14.0
            cell_column = min(int(center_x // CELL_SIZE), GRID_SIZE - 1)
            cell_row = min(int(center_y // CELL_SIZE), GRID_SIZE - 1)

            target_maps[scene_index, 0, cell_row, cell_column] = 1.0                      # objectness
            target_maps[scene_index, 1, cell_row, cell_column] = center_x / CELL_SIZE - cell_column
            target_maps[scene_index, 2, cell_row, cell_column] = center_y / CELL_SIZE - cell_row
            target_maps[scene_index, 3, cell_row, cell_column] = 28.0 / CANVAS_SIZE      # width fraction
            target_maps[scene_index, 4, cell_row, cell_column] = 28.0 / CANVAS_SIZE      # height fraction
            target_maps[scene_index, 5 + digit_class, cell_row, cell_column] = 1.0       # one-hot class
    return scenes, target_maps


class SingleStageDetector(nn.Module):
    """A small conv backbone (64 -> 8 grid) plus a 1x1 prediction head."""

    def __init__(self):
        super().__init__()
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        self.backbone = nn.Sequential(conv_block(1, 16), conv_block(16, 32), conv_block(32, 64))
        # A 1x1 convolution as head: every grid cell makes its own prediction
        # from its own 64 features - detection is per-cell classification+regression.
        self.prediction_head = nn.Conv2d(64, PREDICTION_CHANNELS, kernel_size=1)

    def forward(self, scene_batch):
        return self.prediction_head(self.backbone(scene_batch))


def compute_detection_loss(predictions, target_maps):
    """The three-part single-stage loss.

    Arguments:
        predictions: raw model output (batch, 15, 8, 8).
        target_maps: targets from build_scene_batch, same shape.

    objectness: binary cross-entropy on EVERY cell ("is a center here?");
    box: MSE on the 4 box numbers, only where an object exists;
    class: cross-entropy, only where an object exists.
    """
    objectness_logits = predictions[:, 0]
    object_present = target_maps[:, 0]
    # Only ~2 of 64 cells hold an object, so plain BCE teaches the model that
    # "no" is almost always right and confidence stays low. Weighting the
    # positive cells 5x rebalances the lesson (a standard detector trick).
    objectness_loss = nn.functional.binary_cross_entropy_with_logits(
        objectness_logits, object_present,
        pos_weight=torch.tensor(5.0, device=predictions.device),
    )

    responsible = object_present > 0.5
    if responsible.any():
        # Sigmoid keeps predicted centers inside their cell and sizes in (0,1),
        # matching how the targets were encoded.
        predicted_boxes = torch.sigmoid(predictions[:, 1:5].permute(0, 2, 3, 1)[responsible])
        target_boxes = target_maps[:, 1:5].permute(0, 2, 3, 1)[responsible]
        box_loss = nn.functional.mse_loss(predicted_boxes, target_boxes)

        predicted_classes = predictions[:, 5:].permute(0, 2, 3, 1)[responsible]
        target_classes = target_maps[:, 5:].permute(0, 2, 3, 1)[responsible].argmax(dim=1)
        class_loss = nn.functional.cross_entropy(predicted_classes, target_classes)
    else:
        box_loss = class_loss = torch.tensor(0.0, device=predictions.device)

    return objectness_loss + 5.0 * box_loss + class_loss


def compute_iou(first_box, second_box):
    """Intersection over union of two (x_min, y_min, x_max, y_max) boxes."""
    intersection_width = max(0.0, min(first_box[2], second_box[2]) - max(first_box[0], second_box[0]))
    intersection_height = max(0.0, min(first_box[3], second_box[3]) - max(first_box[1], second_box[1]))
    intersection_area = intersection_width * intersection_height
    first_area = (first_box[2] - first_box[0]) * (first_box[3] - first_box[1])
    second_area = (second_box[2] - second_box[0]) * (second_box[3] - second_box[1])
    return intersection_area / (first_area + second_area - intersection_area + 1e-9)


def decode_predictions(predictions, confidence_threshold=0.4, nms_iou_threshold=0.5):
    """Turn one scene's raw output map into a final list of detections.

    Arguments:
        predictions: (15, 8, 8) raw output for one scene.
        confidence_threshold: minimum objectness probability to keep a cell.
        nms_iou_threshold: overlap above which a weaker box is suppressed.

    Returns a list of (score, class_id, (x_min, y_min, x_max, y_max)).
    """
    candidate_detections = []
    objectness = torch.sigmoid(predictions[0])
    boxes = torch.sigmoid(predictions[1:5])
    class_scores = predictions[5:]
    for cell_row in range(GRID_SIZE):
        for cell_column in range(GRID_SIZE):
            score = float(objectness[cell_row, cell_column])
            if score < confidence_threshold:
                continue
            center_x = (cell_column + float(boxes[0, cell_row, cell_column])) * CELL_SIZE
            center_y = (cell_row + float(boxes[1, cell_row, cell_column])) * CELL_SIZE
            width = float(boxes[2, cell_row, cell_column]) * CANVAS_SIZE
            height = float(boxes[3, cell_row, cell_column]) * CANVAS_SIZE
            box = (center_x - width / 2, center_y - height / 2,
                   center_x + width / 2, center_y + height / 2)
            class_id = int(class_scores[:, cell_row, cell_column].argmax())
            candidate_detections.append((score, class_id, box))

    # Non-maximum suppression: keep the best-scoring box, drop everything that
    # overlaps it too much, repeat. Duplicates arise because neighboring cells
    # can both fire on the same object.
    candidate_detections.sort(key=lambda detection: -detection[0])
    kept_detections = []
    for detection in candidate_detections:
        if all(compute_iou(detection[2], kept[2]) < nms_iou_threshold for kept in kept_detections):
            kept_detections.append(detection)
    return kept_detections


def extract_ground_truth(target_map):
    """Recover the true (class_id, box) list from one scene's target map."""
    truths = []
    for cell_row in range(GRID_SIZE):
        for cell_column in range(GRID_SIZE):
            if target_map[0, cell_row, cell_column] > 0.5:
                center_x = (cell_column + float(target_map[1, cell_row, cell_column])) * CELL_SIZE
                center_y = (cell_row + float(target_map[2, cell_row, cell_column])) * CELL_SIZE
                width = float(target_map[3, cell_row, cell_column]) * CANVAS_SIZE
                height = float(target_map[4, cell_row, cell_column]) * CANVAS_SIZE
                class_id = int(target_map[5:, cell_row, cell_column].argmax())
                truths.append((class_id, (center_x - width / 2, center_y - height / 2,
                                          center_x + width / 2, center_y + height / 2)))
    return truths


def evaluate_detector(model, digit_images, digit_labels, device, scene_count=200):
    """Precision and recall at IoU >= 0.5 with matching class, over fresh scenes."""
    evaluation_generator = torch.Generator().manual_seed(999)
    scenes, target_maps = build_scene_batch(digit_images, digit_labels, scene_count, evaluation_generator)
    model.eval()
    with torch.no_grad():
        predictions = model(scenes.to(device)).cpu()
    model.train()

    true_positives = false_positives = false_negatives = 0
    for scene_index in range(scene_count):
        detections = decode_predictions(predictions[scene_index])
        truths = extract_ground_truth(target_maps[scene_index])
        matched_truths = set()
        for score, class_id, box in detections:
            best_truth = None
            best_iou = 0.5
            for truth_index, (true_class, true_box) in enumerate(truths):
                if truth_index in matched_truths:
                    continue
                overlap = compute_iou(box, true_box)
                if overlap >= best_iou and class_id == true_class:
                    best_iou = overlap
                    best_truth = truth_index
            if best_truth is not None:
                matched_truths.add(best_truth)
                true_positives += 1
            else:
                false_positives += 1
        false_negatives += len(truths) - len(matched_truths)

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    return precision, recall


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--quick", action="store_true", help="600 steps instead of 3000")
    parsed_arguments = argument_parser.parse_args()

    device = select_best_available_device()
    torch.manual_seed(42)
    scene_generator = torch.Generator().manual_seed(42)

    print("Loading MNIST digits to build scenes from...")
    training_dataset, _ = load_mnist_datasets()
    digit_images = training_dataset.data.float() / 255.0
    digit_labels = training_dataset.targets

    model = SingleStageDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    total_steps = 600 if parsed_arguments.quick else 8000

    print(f"Training the detector for {total_steps} steps (fresh synthetic scenes every step)...")
    print("   step    loss    precision  recall   (precision/recall at IoU>=0.5, class must match)")
    for step_number in range(1, total_steps + 1):
        scenes, target_maps = build_scene_batch(digit_images, digit_labels, 32, scene_generator)
        loss = compute_detection_loss(model(scenes.to(device)), target_maps.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_number in (100, total_steps // 4, total_steps // 2, total_steps) or step_number == 1:
            precision, recall = evaluate_detector(model, digit_images, digit_labels, device)
            print(f"  {step_number:>5}   {loss.item():.4f}   {precision:>8.2%}  {recall:>7.2%}")

    print()
    print("Detections on three fresh scenes (threshold 0.25 so near-misses are visible too):")
    demo_generator = torch.Generator().manual_seed(7)
    scenes, target_maps = build_scene_batch(digit_images, digit_labels, 3, demo_generator)
    model.eval()
    with torch.no_grad():
        predictions = model(scenes.to(device)).cpu()
    for scene_index in range(3):
        truths = extract_ground_truth(target_maps[scene_index])
        print(f"  scene {scene_index} truth:     "
              + "  ".join(f"digit {cls} at {tuple(round(v) for v in box)}" for cls, box in truths))
        detections = decode_predictions(predictions[scene_index], confidence_threshold=0.25)
        if not detections:
            print("  scene {0} detected:  (nothing above threshold - a miss)".format(scene_index))
        for score, class_id, box in detections:
            rounded_box = tuple(round(coordinate) for coordinate in box)
            print(f"  scene {scene_index} detected:  digit {class_id} conf {score:.2f} at {rounded_box}")


if __name__ == "__main__":
    main()
