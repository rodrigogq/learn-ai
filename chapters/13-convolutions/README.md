# Chapter 13 — Convolutions

Chapter 9 ended with a diagnosis: the MLP scored "only" 96% on digits because it treats an image as an unordered bag of 784 numbers — shuffle the pixels consistently and it would learn just as well. Images have *structure*: nearby pixels form edges, edges form shapes. The **convolution** is the operation that finally looks at images the way images work, and it powers everything visual from here to the end of the course.

<!-- CONTENTS_START -->
## Contents

- [What you will learn](#what-you-will-learn)
- [Prerequisites](#prerequisites)
- [1. The operation](#1-the-operation)
- [2. Why this beats a dense layer at seeing](#2-why-this-beats-a-dense-layer-at-seeing)
- [3. The four knobs](#3-the-four-knobs)
- [Code walkthrough](#code-walkthrough)
- [Run it](#run-it)
- [What the C version covers](#what-the-c-version-covers)
- [Exercises](#exercises)
- [Next](#next)

<!-- CONTENTS_END -->

## What you will learn

- The convolution operation, worked by hand on paper-sized numbers.
- Kernels as pattern detectors; feature maps as "where the pattern is".
- Padding, stride, channels, and pooling — the four knobs of every CNN layer.
- Why convolutions need ~100× fewer parameters than an MLP for the same job.

## Prerequisites

- [Chapter 9](../09-first-neural-network/README.md) — the MLP whose blindness we cure.
- [Chapter 2](../02-vectors-and-matrices/README.md) — weighted sums (a convolution is many of them).

## 1. The operation

Take a small grid of weights — a **kernel**, typically 3×3 — and slide it across the image. At each position: multiply the patch under the window by the kernel, element by element, and sum. One weighted sum per position; the results form a new grid called a **feature map**.

![A 3x3 vertical-edge kernel sliding over a striped image](figures/sliding-window.svg)

Follow the figure's arithmetic once by hand — it is Chapter 0's weighted sum, applied at every window position. The kernel shown, $(-1, 0, 1)$ in every row, answers one question everywhere: *"is it brighter on my right than on my left?"* On the striped image, its output is +3 along the stripe's left edge, −3 along the right edge, 0 on flat regions. **This kernel is a vertical-edge detector, and its output map says where the edges are.**

Different weights ask different questions: rotate the kernel to detect horizontal edges; other patterns detect corners, blobs, textures. In a CNN, **nobody designs the kernels — the weights are learned by backpropagation**, exactly like every weight so far. Trained networks reliably rediscover edge detectors in their first layer, because edges genuinely are the atoms of images.

## 2. Why this beats a dense layer at seeing

Two structural advantages over Chapter 9's `nn.Linear`:

1. **Locality.** Each output looks at a 3×3 neighborhood, matching how images work: a pixel's meaning depends on its neighbors, not on a pixel in the far corner.
2. **Weight sharing.** The *same* 9 weights are reused at every position. An edge detector is equally valid in the top-left and bottom-right, so learning it once suffices — and a pattern learned from edges appearing anywhere in *any* training image benefits detection *everywhere*. Compare parameter counts for one layer on a 224×224 image: dense, 224²→224² would need ~2.5 *billion* weights; a conv layer with 32 kernels needs about **9 × 32 ≈ 300**.

The price is fair: convolutions assume the pattern's *position* does not change its meaning (true for photos, less true for, say, board games), and each layer sees only locally — which is why CNNs stack many layers, each seeing further than the last (its *receptive field* grows).

## 3. The four knobs

**Padding** — a border of zeros around the image so the kernel can center on edge pixels. Without it, each 3×3 layer shaves a pixel off each side; with `padding=1`, output size = input size ("same" padding).

**Stride** — how far the window jumps per step. Stride 1 visits every position; stride 2 skips every other one, halving the map's width and height. The size formula, verified live by both programs:

$$\text{output size} = \left\lfloor \frac{\text{input size} + 2 \cdot \text{padding} - \text{kernel size}}{\text{stride}} \right\rfloor + 1$$

**Channels** — real images have 3 values per pixel (red, green, blue), so a kernel on an RGB image is really 3×3×**3**, summing over channels too. And each conv layer applies **many** kernels (32, 64, 128…), stacking their feature maps into the next layer's "channels" — layer 2's kernels then combine layer 1's edge maps into corner and texture detectors, and so on up the hierarchy. A CNN's tensor shapes read `(batch, channels, height, width)`.

**Pooling** — shrink a map by summarizing neighborhoods, most commonly *max pooling*: keep the largest value in each 2×2 block. It halves the resolution, keeps the strongest detections, and buys a little position-tolerance. (Modern networks often use stride-2 convolutions instead; same effect, learned.)

A classic CNN is just these pieces repeated — conv, ReLU, pool, repeat — maps getting *smaller* but *deeper* (more channels), until a final dense layer reads the distilled features. Chapter 14 builds exactly that, plus the trick that lets it go really deep.

## Code walkthrough

The example is `python/convolution_from_scratch.py`. The whole chapter lives in **one function**; everything else checks or times it. No prior programming assumed.

### Step 1 — the whole convolution, in one function

```python
def convolve_2d(input_image, kernel, padding=0, stride=1):
    if padding > 0:
        input_image = numpy.pad(input_image, padding)
    input_height, input_width = input_image.shape
    kernel_height, kernel_width = kernel.shape
    output_height = (input_height - kernel_height) // stride + 1
    output_width = (input_width - kernel_width) // stride + 1

    output_map = numpy.zeros((output_height, output_width))
    for output_row in range(output_height):
        for output_column in range(output_width):
            image_patch = input_image[
                output_row * stride: output_row * stride + kernel_height,
                output_column * stride: output_column * stride + kernel_width,
            ]
            output_map[output_row, output_column] = (image_patch * kernel).sum()
    return output_map
```

Read it top to bottom — it is exactly Section 1's "slide and sum":

- `numpy.pad(input_image, padding)` adds the border of zeros (Section 3's *padding* knob), so the kernel can sit on edge pixels.
- The `output_height`/`output_width` lines *are* the size formula from Section 3, in code — how many window positions fit given the kernel size and stride.
- `numpy.zeros((output_height, output_width))` makes the empty feature map to fill in.
- The **two nested `for` loops** visit every output position (row, then column). At each one, `input_image[row : row+kh, column : column+kw]` is a NumPy slice that grabs the little **patch** under the window (`stride` controls how far the window jumps).
- The one line that *is* convolution: `(image_patch * kernel).sum()`. `image_patch * kernel` multiplies the patch by the kernel element by element (NumPy does the whole grid at once), and `.sum()` adds the results — **one weighted sum, exactly Chapter 0**, producing one output pixel. Do that at every position and you have the feature map.

That is the entire operation. Everything below just exercises it.

### Step 2 — the worked example (does it detect edges?)

```python
striped_image = numpy.zeros((5, 5))
striped_image[:, 2:4] = 1.0          # paint two bright columns
output_map = convolve_2d(striped_image, VERTICAL_EDGE_KERNEL)
```

`demonstrate_worked_example` builds the figure's striped image (`[:, 2:4] = 1.0` sets columns 2–3 to bright) and runs the `(-1, 0, 1)` vertical-edge kernel over it. The printed output is +3 where brightness rises and −3 where it falls — the figure's exact numbers, confirming the kernel really is an edge detector.

### Step 3 — padding and stride, against the formula

```python
for padding, stride in ((0, 1), (1, 1), (1, 2), (0, 2)):
    output_map = convolve_2d(test_image, VERTICAL_EDGE_KERNEL, padding, stride)
    formula_size = (28 + 2 * padding - 3) // stride + 1
```

`demonstrate_padding_and_stride` runs four padding/stride combinations on a 28×28 image and prints the actual output shape next to the Section 3 formula's prediction — they match. You can watch `padding=1` keep the size at 28×28 ("same" padding) and `stride=2` halve it to 14×14, which is how CNNs shrink their maps.

### Step 4 — is it *correct*, and how slow?

`demonstrate_agreement_with_pytorch` runs our loops and PyTorch's real `torch.nn.functional.conv2d` on the same image and reports the largest difference — about 1e-15, i.e. identical to floating-point precision. That is the true correctness check: our from-scratch version computes exactly what the framework does. Then `demonstrate_speed` times both on a 224×224 image, setting up the chapter's punchline — on one small single-channel image, plain compiled C loops match PyTorch; the framework's real edge is batched, many-channel workloads on a GPU.

### Quick reference

| Function | What it does | What to notice |
|----------|--------------|----------------|
| `convolve_2d(image, kernel, padding, stride)` | **The whole operation** — pad, then slide the kernel computing `(patch × kernel).sum()`. | The core line is one weighted sum (Chapter 0!) per output pixel. |
| `demonstrate_worked_example()` | The vertical-edge kernel on the striped image. | Output is +3 / −3 along the edges — the figure's exact numbers. |
| `demonstrate_padding_and_stride()` | Output sizes for four padding/stride combos vs the formula. | `padding=1` keeps size (28→28); `stride=2` halves it. |
| `demonstrate_agreement_with_pytorch()` | Compares against `torch.nn.functional.conv2d`. | Difference ~1e-15 — the real correctness check. |
| `demonstrate_speed()` | Times Python loops vs PyTorch on a 224×224 image. | Sets up the punchline; C lands close to PyTorch on this single-channel case. |

**Carry forward:** `convolve_2d` is the operation behind every vision chapter. Chapter 14's C ResNet calls a multi-channel version of this exact loop.

## Run it

```bash
.venv/bin/python chapters/13-convolutions/python/convolution_from_scratch.py
make -C chapters/13-convolutions/c && ./chapters/13-convolutions/c/build/convolution_benchmark
```

The Python program: the figure's example, the size formula live, an exact-agreement check against `torch.nn.functional.conv2d` (differences ~1e-15), and a timing. The C program: the same example and the same 224×224 workload, timed.

Numbers from the reference machine — the punchline is worth staring at:

```
Python loops: 32.1 ms      PyTorch: 0.06 ms      C loops (-O2): 0.058 ms
```

On one small single-channel image, **60 lines of plain C match PyTorch** — the "magic" of fast frameworks is largely just compiled loops. PyTorch pulls ahead where it counts: batched, many-channel workloads, and above all on GPUs, where thousands of those weighted sums run at once (convolution is embarrassingly parallel — every output pixel is independent).

## What the C version covers

A full port of the core algorithm plus the padding helper and the benchmark. Note how small it is: convolution needs no framework, no allocation tricks, nothing — four nested loops and the flat-index formula from Chapter 2.

## Exercises

1. By hand: apply the horizontal-edge kernel (the figure's kernel, transposed) to the striped image. Predict the output before computing: where are the *horizontal* edges in vertical stripes?
2. In the Python file, build a 3×3 kernel of all 1/9 values and convolve any image. What everyday image-editor operation did you just implement?
3. Using the size formula: what padding keeps a 5×5 kernel "same"? A 7×7? State the general rule for odd kernel sizes.
4. Stack two 3×3 convolutions (convolve the output again). What is the receptive field of one output value — how many input pixels influence it? This is why deep stacks of small kernels replaced single big kernels.
5. Challenge (C): extend `convolve_2d` to multi-channel input (a `channel_count` parameter and a 3D kernel) — the sum gains one more loop. Verify against PyTorch with a 3-channel image.

## Next

[Chapter 14 — Image classification](../14-image-classification/README.md)

<!-- NAV_START -->
---

[← Chapter 12: Data pipelines](../12-data-pipelines/README.md) · [↑ Course index](../../README.md) · [Chapter 14: Image classification →](../14-image-classification/README.md)

<!-- NAV_END -->
