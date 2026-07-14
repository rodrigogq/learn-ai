# Chapter 15 — Object detection

Classification answers *what*; detection answers *what and where* — for every object in the scene at once. This is the chapter of bounding boxes, and of the three ideas that make box-prediction work: **IoU** (measuring box agreement), **grid-based prediction** (every image region votes), and **NMS** (cleaning up duplicate votes). You will train a real single-stage detector — the same design family as YOLO — on scenes it can never memorize, because every training image is generated fresh.

<!-- CONTENTS_START -->
## Contents

- [What you will learn](#what-you-will-learn)
- [Prerequisites](#prerequisites)
- [1. The task, and the dataset trick](#1-the-task-and-the-dataset-trick)
- [2. IoU: the ruler for boxes](#2-iou-the-ruler-for-boxes)
- [3. The single-stage detector](#3-the-single-stage-detector)
- [4. Scoring a detector](#4-scoring-a-detector)
- [Code walkthrough](#code-walkthrough)
- [Run it](#run-it)
- [What the C version covers](#what-the-c-version-covers)
- [Exercises](#exercises)
- [Next](#next)

<!-- CONTENTS_END -->

## What you will learn

- The detection task and how boxes are represented.
- IoU — intersection over union — worked by hand.
- The single-stage recipe: a grid of cells, each predicting objectness + box + class.
- Non-maximum suppression, and why detectors need it.
- Honest detection metrics: precision and recall at an IoU threshold.

## Prerequisites

- [Chapter 14](../14-image-classification/README.md) — conv backbones.
- [Chapter 12](../12-data-pipelines/README.md) — precision and recall.

## 1. The task, and the dataset trick

A detector receives an image and returns a *list*: for each object, a class label, a **bounding box** (here as its four corner coordinates `(x_min, y_min, x_max, y_max)` — left, top, right, bottom), and a confidence score. Lists are awkward for networks — variable length, no natural order — and the whole design of this chapter is about turning "predict a list" into "fill in a fixed grid".

The dataset is manufactured, and that is a feature: each 64×64 training scene is composed on the fly by pasting 1–3 random MNIST digits at random positions, so the ground-truth boxes are known *exactly* (we placed them) and the supply of scenes is infinite (Chapter 12's augmentation idea taken to its limit). Real detection datasets — COCO is the standard — work the same way conceptually, with humans drawing the boxes instead.

## 2. IoU: the ruler for boxes

Any claim like "the predicted box is right" needs a number. **Intersection over union** divides the overlap area of two boxes by the area of their combined footprint:

![IoU worked example and non-maximum suppression before and after](figures/iou-and-nms.svg)

Work the figure's example by hand once: boxes A = (20,20)–(60,60) and B = (40,30)–(80,70) intersect in a 20×30 = 600 rectangle; the union is 1600 + 1600 − 600 = 2600; IoU = 600/2600 ≈ **0.23**. The scale is intuitive: 1.0 is a perfect match, 0 is disjoint, and the field's standard bar for "close enough to count" is **IoU ≥ 0.5**.

## 3. The single-stage detector

The design that turned detection into one forward pass (the YOLO family, 2016 onward):

1. A small conv backbone (Chapter 14's kind) shrinks the 64×64 image to an **8×8 grid** of feature vectors — each cell effectively summarizes an 8×8 patch of the image.
2. A 1×1 convolution head makes every cell predict 15 numbers: **1 objectness score** ("does an object's *center* fall in my patch?"), **4 box numbers** (center offset within the cell via sigmoid, width and height as fractions of the image), and **10 class scores**.
3. Training assigns each ground-truth digit to the one cell containing its center; that cell is "responsible" for it.

The loss has three parts, one per prediction type: binary cross-entropy on objectness over *all* cells, box regression (MSE) and class cross-entropy only on responsible cells. One imbalance needs handling — only ~2 of 64 cells contain an object, so unweighted BCE teaches the network that "no" is almost always correct and its confidence never rises; weighting positive cells 5× (the code's `pos_weight`) rebalances the lesson. Class imbalance is *the* recurring headache of detection — real detectors fight it with focal loss and hard-negative mining, both descendants of this same fix.

At inference, cells above a confidence threshold each emit a box — and neighboring cells routinely fire on the same object, which is not a bug (their patches genuinely both contain it) but leaves duplicates. **Non-maximum suppression** cleans up: keep the highest-confidence box, delete everything overlapping it with IoU > 0.5, repeat. Three lines of algorithm, run by every detector on earth; the figure shows it collapsing three candidate boxes into one.

## 4. Scoring a detector

Accuracy makes no sense for lists. Instead, match each detection to an unmatched ground-truth object (same class, IoU ≥ 0.5): matched detections are **true positives**, unmatched detections **false positives** (hallucinated boxes), unmatched truths **false negatives** (missed objects) — and Chapter 12's precision and recall apply verbatim. The training run reports both:

```
   step    loss    precision  recall   (precision/recall at IoU>=0.5, class must match)
      1   3.8976      0.61%    9.47%
    100   2.0531     23.18%   34.54%
   2000   1.0093     62.79%   67.69%
   8000   1.0303     75.62%   76.88%

Detections on three fresh scenes (threshold 0.25 so near-misses are visible too):
  scene 0 truth:     digit 5 at (2, 25, 30, 53)
  scene 0 detected:  digit 1 conf 0.40 at (2, 26, 30, 54)      <- perfect box, wrong class
  scene 1 truth:     digit 8 at (34, 17, 62, 45)  digit 9 at (23, 35, 51, 63)
  scene 1 detected:  digit 9 conf 0.99 at (23, 35, 51, 63)     <- exact
  scene 1 detected:  digit 8 conf 0.76 at (36, 19, 64, 47)
  scene 1 detected:  digit 2 conf 0.48 at (31, 28, 59, 56)     <- a ghost between two real digits
```

The failure modes on display are the real ones: a pixel-perfect box carrying the wrong class (scene 0 — the digit was half cut off at the image edge), and a "ghost" detection hallucinated *between* two overlapping objects (scene 1) — the kind NMS cannot remove because it overlaps neither real box enough.

Reading the trade-off: raise the confidence threshold and precision climbs while recall falls; lower it and the reverse. Real benchmarks sweep the threshold and integrate the whole curve into **mAP** (mean average precision, averaged over classes and IoU thresholds) — the number in every detection paper. Our single-threshold precision/recall is mAP's honest little sibling.

Also visible in the results: detection is genuinely harder than classification. The same digits that Chapter 9 classified at 96% yield much lower detection scores — the model must *find* them, *box* them, and classify them, with overlapping digits and truncated context. State-of-the-art detectors close the gap with anchors of multiple sizes, multi-scale feature pyramids, and much bigger backbones — extensions, not different ideas.

## Code walkthrough

The example is `python/train_digit_detector.py`. Detection has more moving parts than a classifier, so read it as a pipeline: build data → predict → score. No prior programming assumed.

### Step 1 — turning "predict a list" into "fill in a grid"

```python
cell_column = min(int(center_x // CELL_SIZE), GRID_SIZE - 1)
cell_row = min(int(center_y // CELL_SIZE), GRID_SIZE - 1)
target_maps[scene_index, 0, cell_row, cell_column] = 1.0                    # objectness
target_maps[scene_index, 1, cell_row, cell_column] = center_x / CELL_SIZE - cell_column
...
target_maps[scene_index, 5 + digit_class, cell_row, cell_column] = 1.0      # one-hot class
```

`build_scene_batch` pastes 1–3 random digits onto a blank 64×64 canvas — and, crucially, builds the **target grid** the network will be trained to reproduce. For each pasted digit it finds which of the 8×8 cells holds the digit's *center*, and writes into that cell: a `1.0` objectness flag ("an object's center is here"), the 4 box numbers, and a one-hot class. This is the whole idea of the chapter — a variable-length *list* of objects becomes a fixed-size grid the network can output. The target is laid out in exactly the channel order the model predicts, which is what makes the loss simple.

### Step 2 — the model: a backbone then a 1×1 head

```python
self.backbone = nn.Sequential(conv_block(1, 16), conv_block(16, 32), conv_block(32, 64))
self.prediction_head = nn.Conv2d(64, PREDICTION_CHANNELS, kernel_size=1)
```

The backbone is three stride-2 conv blocks (Chapter 14's kind) that shrink 64×64 down to an 8×8 grid of 64-feature vectors. The head is a **1×1 convolution** — the elegant trick: a 1×1 conv applies the same little classifier independently to every grid cell, turning each cell's 64 features into its 15 predictions. Detection is just per-cell classify-and-regress, run 64 times in one convolution.

### Step 3 — the three-part loss (and the imbalance fix)

```python
objectness_loss = nn.functional.binary_cross_entropy_with_logits(
    objectness_logits, object_present,
    pos_weight=torch.tensor(5.0, device=predictions.device),
)
responsible = object_present > 0.5
...
return objectness_loss + 5.0 * box_loss + class_loss
```

Three losses added up: **objectness** (is a center here?) scored on *every* cell with binary cross-entropy; **box** (MSE) and **class** (cross-entropy) scored only on the `responsible` cells — the ones that actually contain an object. The key line is `pos_weight=5.0`: since only ~2 of 64 cells hold an object, plain BCE would let the model win by always saying "no". Weighting the rare positive cells 5× rebalances the lesson — Section 3's fix, and the ancestor of focal loss.

### Step 4 — IoU, the ruler for boxes

```python
intersection_width = max(0.0, min(first_box[2], second_box[2]) - max(first_box[0], second_box[0]))
intersection_height = max(0.0, min(first_box[3], second_box[3]) - max(first_box[1], second_box[1]))
intersection_area = intersection_width * intersection_height
return intersection_area / (first_area + second_area - intersection_area + 1e-9)
```

`compute_iou` is Section 2 in code: the overlap rectangle's area divided by the two boxes' combined area. The `max(0.0, ...)` handles non-overlapping boxes (a negative width becomes 0). This one small function is reused everywhere below — by NMS and by scoring.

### Step 5 — decoding: threshold, then non-maximum suppression

```python
candidate_detections.sort(key=lambda detection: -detection[0])   # strongest first
kept_detections = []
for detection in candidate_detections:
    if all(compute_iou(detection[2], kept[2]) < nms_iou_threshold for kept in kept_detections):
        kept_detections.append(detection)
```

`decode_predictions` first turns every cell above the confidence threshold into a box (undoing the encoding from Step 1), then runs **NMS** to remove duplicates from neighboring cells that fired on the same object. The loop is the whole algorithm: sort candidates strongest-first, then keep a box only if it does *not* overlap an already-kept box by more than the threshold. "Keep the best, drop its overlaps, repeat" — three lines, run by every detector on earth.

### Step 6 — scoring: matching detections to truths

`evaluate_detector` matches each detection to an unmatched ground-truth object of the same class with IoU ≥ 0.5. Matched detections are **true positives**, unmatched detections **false positives**, unmatched truths **false negatives** — and then Chapter 12's precision and recall apply verbatim. This is why accuracy makes no sense for detection: there is no single "right answer" per image, only a set of boxes to match up.

The C file `c/iou_and_nms.c` is `compute_iou` and NMS in pure C — the exact post-processing that runs inside every deployed detector, including your phone's camera.

### Quick reference

| Function | What it does | What to notice |
|----------|--------------|----------------|
| `build_scene_batch(...)` | Pastes digits and builds the **target grid**: objectness + box + class at each digit's center cell. | The target layout mirrors the model output — that alignment makes the loss simple. |
| `class SingleStageDetector` | Conv backbone (64→8 grid) + a 1×1 head predicting 15 numbers per cell. | The **1×1 conv** head makes every cell predict independently. |
| `compute_detection_loss(...)` | Objectness (all cells) + box + class (object cells only). | `pos_weight=5` fixes the ~2-of-64 imbalance. |
| `compute_iou(box_a, box_b)` | Intersection over union. | The universal ruler — reused by NMS and scoring. |
| `decode_predictions(...)` | Threshold cells, then NMS to drop duplicates. | The greedy keep-best-drop-overlaps loop is right here. |
| `evaluate_detector(...)` | Match detections to truths, compute precision/recall. | How detection is actually scored. |

## Run it

```bash
.venv/bin/python chapters/15-object-detection/python/train_digit_detector.py --quick   # ~1 min
.venv/bin/python chapters/15-object-detection/python/train_digit_detector.py           # ~8 min

make -C chapters/15-object-detection/c && ./chapters/15-object-detection/c/build/iou_and_nms
```

## What the C version covers

IoU and NMS, complete — reproducing the chapter's worked example (0.231) and running suppression on five candidate detections (three duplicates of one digit collapse to the strongest; a distant weak box correctly survives, because NMS removes *duplicates*, not low-confidence detections — that is the threshold's job). These ~60 lines are exactly the post-processing that runs inside every deployed detector, including the ones in your phone's camera.

## Exercises

1. By hand: compute the IoU of (0,0)–(10,10) and (5,5)–(15,15). Then of (0,0)–(10,10) and (10,0)–(20,10) — boxes that share only an edge.
2. In the training script, sweep the decode threshold (0.2 / 0.4 / 0.6 / 0.8) on the final model and tabulate precision vs recall. You have hand-computed a precision-recall curve.
3. Break NMS: set `nms_iou_threshold=0.05` and describe what happens when two *different* digits sit close together. Now set 0.95 — what comes back?
4. The center-cell assignment fails when two digits' centers fall in the same cell. Estimate how often that happens with 3 digits on an 8×8 grid, then check: print a warning in `build_scene_batch` when it occurs. (Real YOLO assigns multiple "anchors" per cell partly for this.)
5. Challenge: add a fourth digit to every scene and retrain. Which metric suffers more, precision or recall, and why? (Think about what crowding does to the center-cell scheme.)

## Next

[Chapter 16 — Segmentation](../16-segmentation/README.md)

<!-- NAV_START -->
---

[← Chapter 14: Image classification](../14-image-classification/README.md) · [↑ Course index](../../README.md) · [Chapter 16: Segmentation →](../16-segmentation/README.md)

<!-- NAV_END -->
