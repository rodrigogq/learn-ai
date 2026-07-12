# Appendix B — Python refresher

The course assumes you can already read and write basic Python. This appendix reviews the specific features the course leans on, so nothing in the chapters surprises you. It is a checklist, not a Python course — if most of this looks new, spend a day with the [official Python tutorial](https://docs.python.org/3/tutorial/) first.

## Lists, loops, and comprehensions

```python
squared_values = []
for value in [1, 2, 3]:
    squared_values.append(value * value)

# The same thing as a "list comprehension" - the course uses these often:
squared_values = [value * value for value in [1, 2, 3]]
```

## Functions, default arguments, keyword arguments

```python
def scale_vector(vector, scale_factor=2.0):
    """Multiply every element of `vector` by `scale_factor`."""
    return [element * scale_factor for element in vector]

scale_vector([1.0, 2.0])                      # uses the default 2.0
scale_vector([1.0, 2.0], scale_factor=10.0)   # keyword argument, order-free
```

Every function in this course starts with a docstring explaining what each argument means, like above.

## Classes (used from Chapter 8 on)

```python
class Neuron:
    """One artificial neuron holding its own weights."""

    def __init__(self, number_of_inputs):
        self.weights = [0.0] * number_of_inputs
        self.bias = 0.0

    def compute_output(self, input_values):
        """Weighted sum of the inputs plus the bias."""
        weighted_sum = sum(weight * value for weight, value in zip(self.weights, input_values))
        return weighted_sum + self.bias
```

`self` is the instance the method was called on. `__init__` runs when the object is created. PyTorch models (Chapter 10 on) are classes that inherit from `torch.nn.Module` — the chapters explain that pattern when it appears.

## Tuples and multiple return values

```python
def split_dataset(examples):
    """Return (training_part, test_part)."""
    split_point = int(len(examples) * 0.8)
    return examples[:split_point], examples[split_point:]

training_examples, test_examples = split_dataset(all_examples)
```

## Slicing

```python
values = [10, 20, 30, 40, 50]
values[1:3]    # [20, 30]  - from index 1 up to (not including) 3
values[:2]     # [10, 20]  - first two
values[-1]     # 50        - last element
values[::2]    # [10, 30, 50] - every second element
```

NumPy arrays (Chapter 2 on) extend this to several dimensions: `matrix[0, :]` is "row 0, all columns".

## f-strings

```python
loss_value = 0.03456
print(f"epoch {epoch_number}: loss = {loss_value:.4f}")   # loss = 0.0346
```

`:.4f` means "format as a decimal with 4 digits after the point". Training scripts print like this constantly.

## `if __name__ == "__main__"`

```python
def main():
    ...

if __name__ == "__main__":
    main()
```

The block runs only when the file is executed directly (`python file.py`), not when it is imported. Every runnable example in the course uses this pattern.

## `argparse` (used by the longer training scripts)

```python
import argparse

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--quick", action="store_true", help="run a tiny version in seconds")
argument_parser.add_argument("--learning-rate", type=float, default=0.01)
parsed_arguments = argument_parser.parse_args()
```

This is how scripts get flags like `--quick` and `--resume` from the command line.

## Paths with `pathlib`

```python
from pathlib import Path

checkpoint_directory = Path("checkpoints")
checkpoint_directory.mkdir(exist_ok=True)
checkpoint_file_path = checkpoint_directory / "latest.ckpt"   # "/" joins paths
if checkpoint_file_path.exists():
    ...
```

## What you do NOT need

No decorators, generators, async, metaclasses, or type-system tricks. When a chapter needs anything beyond this page, it explains it on the spot.
