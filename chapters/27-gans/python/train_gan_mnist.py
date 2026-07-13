"""Chapter 27 - a GAN: a forger and a detective trained against each other.

Two networks play a game on MNIST:
  - the GENERATOR turns random noise into fake digit images;
  - the DISCRIMINATOR judges images as real (from the dataset) or fake.
They train together: the discriminator learns to catch fakes, the generator
learns to fool it. At equilibrium the fakes are indistinguishable from real -
and that generator is our image factory.

This is a small convolutional GAN (DCGAN-style). It prints ASCII samples as it
trains so you can watch the forgeries improve from static to digits.

Run from the repository root:
    .venv/bin/python chapters/27-gans/python/train_gan_mnist.py --quick
    .venv/bin/python chapters/27-gans/python/train_gan_mnist.py
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from common.data import load_mnist_datasets  # noqa: E402
from common.device import select_best_available_device  # noqa: E402

NOISE_SIZE = 64      # the generator's random input: its "seed" for one image


class Generator(nn.Module):
    """Noise (64) -> a 28x28 image, via transposed convolutions that grow the
    spatial size 7 -> 14 -> 28 (Chapter 16's upsampling, reused)."""

    def __init__(self):
        super().__init__()
        self.project = nn.Linear(NOISE_SIZE, 128 * 7 * 7)
        self.network = nn.Sequential(
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),  # 7->14
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),                                    # 14->28
            nn.Sigmoid(),   # pixels in 0..1
        )

    def forward(self, noise):
        projected = self.project(noise).view(-1, 128, 7, 7)
        return self.network(projected)


class Discriminator(nn.Module):
    """A 28x28 image -> one number: probability it is REAL. An ordinary CNN
    classifier (Chapter 14), just with a single output."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2),       # 28->14
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),  # 14->7
            nn.Flatten(), nn.Linear(128 * 7 * 7, 1),
        )

    def forward(self, images):
        return self.network(images)   # a raw logit; loss applies the sigmoid


def render_digit_ascii(pixels):
    """A 28x28 image tensor as a 14x14 character grid."""
    grid = pixels.detach().cpu().reshape(28, 28)
    characters = " .:-=+*#%@"
    lines = []
    for row in range(0, 28, 2):
        line = "".join(characters[min(int(grid[row, column].item() * len(characters)), len(characters) - 1)]
                        for column in range(0, 28, 2))
        lines.append("   " + line)
    return "\n".join(lines)


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--quick", action="store_true", help="1 epoch instead of 8")
    parsed_arguments = argument_parser.parse_args()

    device = select_best_available_device()
    torch.manual_seed(42)
    epochs = 1 if parsed_arguments.quick else 8

    training_dataset, _ = load_mnist_datasets()  # 2D images, not flattened
    training_loader = DataLoader(training_dataset, batch_size=128, shuffle=True)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    binary_loss = nn.BCEWithLogitsLoss()

    # A fixed noise vector, sampled once, so the printed samples across epochs
    # show the SAME seed improving - the clearest way to watch progress.
    fixed_noise = torch.randn(1, NOISE_SIZE, device=device)

    print("Training the forger (generator) against the detective (discriminator)...")
    print("Watch the fixed-seed sample turn from static into a digit.\n")
    for epoch in range(1, epochs + 1):
        for real_images, _ in training_loader:
            real_images = real_images.to(device)
            batch_size = real_images.shape[0]
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # --- train the discriminator: real should score 1, fake 0 ---
            noise = torch.randn(batch_size, NOISE_SIZE, device=device)
            fake_images = generator(noise)
            discriminator_loss = (
                binary_loss(discriminator(real_images), real_labels)
                + binary_loss(discriminator(fake_images.detach()), fake_labels)  # detach: don't train G here
            )
            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # --- train the generator: make the discriminator say "real" (1) ---
            # The generator's goal is the OPPOSITE label for its own fakes -
            # that adversarial flip is the whole game.
            generator_loss = binary_loss(discriminator(fake_images), real_labels)
            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

        generator.eval()
        with torch.no_grad():
            sample = generator(fixed_noise)[0, 0]
        generator.train()
        print(f"epoch {epoch}/{epochs}  (D loss {discriminator_loss.item():.2f}, G loss {generator_loss.item():.2f})")
        print(render_digit_ascii(sample))
        print()

    print("The generator now forges digits from pure noise - a working image factory,")
    print("trained without ever being shown a single label saying 'this is how a 3 looks'.")


if __name__ == "__main__":
    main()
