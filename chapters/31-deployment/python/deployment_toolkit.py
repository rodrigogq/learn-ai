"""Chapter 31 - the deployment toolkit: measure, shrink, and export a model.

A trained model is not a product until it runs where it is needed - fast
enough, small enough, on the right hardware. This script demonstrates the
core deployment moves on the Chapter 9 MNIST network (small enough to be
fast, real enough to be honest):
  1. measure the baseline: size on disk and inference latency,
  2. dynamic int8 quantization: ~4x smaller, usually faster on CPU,
  3. TorchScript export: a portable, Python-free model file,
  4. a latency/size/accuracy comparison table - the numbers a deployment
     decision actually turns on.

Run from the repository root:
    .venv/bin/python chapters/31-deployment/python/deployment_toolkit.py
"""

import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from common.data import load_mnist_datasets  # noqa: E402

CHECKPOINT_DIRECTORY = REPOSITORY_ROOT / "checkpoints"


class DigitClassifier(nn.Module):
    """Chapter 9's 784 -> 128 -> 10 network - our deployment specimen."""

    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(784, 128)
        self.output_layer = nn.Linear(128, 10)

    def forward(self, images):
        return self.output_layer(torch.relu(self.hidden_layer(images)))


def quantize_linears_to_int8(fresh_model, trained_model):
    """Round each Linear layer's weights to int8 with a per-tensor scale.

    Arguments:
        fresh_model: a same-shape untrained model to receive the quantized weights.
        trained_model: the float32 model whose weights we quantize.

    Returns fresh_model, its Linear weights replaced by dequantized int8
    values (int8 * scale). Biases stay float32 - they are tiny and sensitive.
    This is Chapter 25's scheme, here proving the size/accuracy trade on a
    real classifier.
    """
    fresh_model.load_state_dict(trained_model.state_dict())
    with torch.no_grad():
        for module in fresh_model.modules():
            if isinstance(module, nn.Linear):
                scale = module.weight.abs().max() / 127.0
                quantized = torch.round(module.weight / scale).clamp(-127, 127)
                module.weight.copy_(quantized * scale)   # store the dequantized values
                module._int8_scale = float(scale)         # remember the scale for sizing
    return fresh_model


def save_int8_state(model, path):
    """Save the quantized model at its true int8 footprint (1 byte per weight)."""
    payload = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            scale = getattr(module, "_int8_scale", module.weight.abs().max().item() / 127.0)
            payload[name + ".weight_int8"] = torch.round(module.weight / scale).to(torch.int8)
            payload[name + ".scale"] = torch.tensor(scale)
            payload[name + ".bias"] = module.bias
    torch.save(payload, path)


def train_briefly(model, training_loader, device):
    """A quick train so the accuracy numbers are meaningful (not the point here)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()
    model.to(device)
    for _ in range(2):
        for images, labels in training_loader:
            loss = loss_function(model(images.to(device)), labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.cpu()


def measure_accuracy(model, test_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            correct += (model(images).argmax(dim=1) == labels).sum().item()
            total += len(labels)
    return correct / total


def measure_latency(model, sample_batch, repeats=200):
    """Average milliseconds per inference on a single image (CPU)."""
    model.eval()
    with torch.no_grad():
        for _ in range(10):        # warm up
            model(sample_batch)
        start = time.perf_counter()
        for _ in range(repeats):
            model(sample_batch)
        return 1000.0 * (time.perf_counter() - start) / repeats


def file_size_kilobytes(path):
    return path.stat().st_size / 1024


def main():
    CHECKPOINT_DIRECTORY.mkdir(exist_ok=True)
    print("Loading MNIST and training a small classifier to deploy...")
    training_dataset, test_dataset = load_mnist_datasets(flatten_images_to_vectors=True)
    training_loader = DataLoader(training_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    # Deployment is a CPU story: servers and edge devices run inference on CPUs
    # far more often than GPUs, and quantization targets CPU int8 paths.
    model = train_briefly(DigitClassifier(), training_loader, torch.device("cpu"))
    sample_batch = test_dataset[0][0][None]

    print("\n1. Baseline (float32)")
    float_path = CHECKPOINT_DIRECTORY / "deploy_float32.pt"
    torch.save(model.state_dict(), float_path)
    float_accuracy = measure_accuracy(model, test_loader)
    float_latency = measure_latency(model, sample_batch)
    float_size = file_size_kilobytes(float_path)
    print(f"   size {float_size:6.1f} KB   latency {float_latency:.3f} ms   accuracy {float_accuracy:.2%}")

    print("\n2. int8 quantization (Chapter 25's method: int8 weights + a scale)")
    # PyTorch's built-in quantize_dynamic needs a CPU backend that is not
    # present in every build, so we quantize the weights ourselves - exactly
    # the per-tensor int8 scheme from Chapter 25, applied to the Linear layers.
    # It is transparent, portable, and makes the size story concrete.
    quantized_model = quantize_linears_to_int8(DigitClassifier(), model)
    quantized_path = CHECKPOINT_DIRECTORY / "deploy_int8.pt"
    save_int8_state(quantized_model, quantized_path)
    quantized_accuracy = measure_accuracy(quantized_model, test_loader)
    quantized_latency = measure_latency(quantized_model, sample_batch)
    quantized_size = file_size_kilobytes(quantized_path)
    print(f"   size {quantized_size:6.1f} KB   latency {quantized_latency:.3f} ms   accuracy {quantized_accuracy:.2%}")

    print("\n3. TorchScript export (a portable, Python-free model file)")
    # TorchScript freezes the model into a self-contained file that C++ / mobile
    # runtimes load without any Python. This is what actually ships.
    scripted = torch.jit.script(model)
    scripted_path = CHECKPOINT_DIRECTORY / "deploy_scripted.pt"
    scripted.save(str(scripted_path))
    reloaded = torch.jit.load(str(scripted_path))
    print(f"   saved {scripted_path.name} ({file_size_kilobytes(scripted_path):.1f} KB), "
          f"reloaded with no model code, accuracy {measure_accuracy(reloaded, test_loader):.2%}")
    print("   (ONNX is the cross-framework alternative - one export, many runtimes.)")

    print("\n4. The deployment decision table")
    print("   variant        size      latency    accuracy")
    print(f"   float32     {float_size:6.1f} KB   {float_latency:6.3f} ms   {float_accuracy:.2%}")
    print(f"   int8        {quantized_size:6.1f} KB   {quantized_latency:6.3f} ms   {quantized_accuracy:.2%}")
    print(f"   -> int8 is {float_size / quantized_size:.1f}x smaller for a "
          f"{abs(float_accuracy - quantized_accuracy) * 100:.2f} point accuracy change.")
    print("   Real deployment is choosing a point on this trade-off surface for your")
    print("   constraints - phone vs server, battery vs speed, size vs the last 0.1%.")


if __name__ == "__main__":
    main()
