"""Chapter 26 - autoencoders and variational autoencoders on MNIST.

Two models, one script:
  1. a plain AUTOENCODER: squeeze each digit through a 2-number bottleneck and
     reconstruct it - showing that 784 pixels really live on a low-dimensional
     surface, and that the bottleneck learns a meaningful code;
  2. a VARIATIONAL autoencoder (VAE): the same idea made GENERATIVE by forcing
     the latent codes into a smooth, sampleable distribution - so we can draw
     a random point and decode a brand-new digit that was never in the data.

The 2-D latent space is deliberately tiny so the script can print an ASCII
map of it: where each digit lives, and a grid of digits decoded from points
we chose, never trained on.

Run from the repository root:
    .venv/bin/python chapters/26-autoencoders-and-vaes/python/train_vae_mnist.py --quick
    .venv/bin/python chapters/26-autoencoders-and-vaes/python/train_vae_mnist.py
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

LATENT_SIZE = 2      # tiny on purpose, so we can visualize the whole latent space


class Autoencoder(nn.Module):
    """Encoder squeezes 784 -> 2; decoder expands 2 -> 784. No randomness."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, LATENT_SIZE),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_SIZE, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid(),   # pixels in 0..1
        )

    def forward(self, images):
        latent_code = self.encoder(images)
        return self.decoder(latent_code), latent_code


class VariationalAutoencoder(nn.Module):
    """The generative upgrade: the encoder outputs a DISTRIBUTION per image
    (a mean and a spread), we sample from it, and a second loss term pulls all
    those distributions toward a standard normal - making the latent space
    smooth and sampleable."""

    def __init__(self):
        super().__init__()
        self.encoder_trunk = nn.Sequential(
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 64), nn.ReLU(),
        )
        self.latent_mean = nn.Linear(64, LATENT_SIZE)
        self.latent_log_variance = nn.Linear(64, LATENT_SIZE)
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_SIZE, 64), nn.ReLU(),
            nn.Linear(64, 256), nn.ReLU(),
            nn.Linear(256, 784), nn.Sigmoid(),
        )

    def encode(self, images):
        features = self.encoder_trunk(images)
        return self.latent_mean(features), self.latent_log_variance(features)

    def forward(self, images):
        mean, log_variance = self.encode(images)
        # The reparameterization trick: sample = mean + std * noise. Writing it
        # this way keeps the randomness OUT of the path gradients flow through
        # (the noise is a constant input), so backprop still works - the one
        # clever idea that makes VAEs trainable.
        standard_deviation = torch.exp(0.5 * log_variance)
        sampled_code = mean + standard_deviation * torch.randn_like(standard_deviation)
        return self.decoder(sampled_code), mean, log_variance


def vae_loss(reconstruction, images, mean, log_variance):
    """Reconstruction loss + the KL term that shapes the latent space.

    Arguments:
        reconstruction, images: decoded and original pixels (batch, 784).
        mean, log_variance: the encoder's per-image distribution parameters.

    reconstruction: binary cross-entropy, "did the pixels come back?";
    KL divergence: a closed-form penalty measuring how far each image's
        latent distribution is from a standard normal - this is what forces
        the codes to fill the space smoothly instead of scattering.
    """
    reconstruction_loss = nn.functional.binary_cross_entropy(reconstruction, images, reduction="sum")
    kl_divergence = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())
    return (reconstruction_loss + kl_divergence) / images.shape[0]


def render_digit_ascii(pixels):
    """A 28x28 pixel vector as a compact 14x14 character grid."""
    grid = pixels.reshape(28, 28)
    characters = " .:-=+*#%@"
    lines = []
    for row in range(0, 28, 2):
        line = "".join(characters[min(int(grid[row, column] * len(characters)), len(characters) - 1)]
                        for column in range(0, 28, 2))
        lines.append("   " + line)
    return "\n".join(lines)


def train_model(model, is_variational, training_loader, device, epochs):
    """Shared training loop for both models."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batch_count = 0
        for images, _ in training_loader:
            images = images.to(device)
            if is_variational:
                reconstruction, mean, log_variance = model(images)
                loss = vae_loss(reconstruction, images, mean, log_variance)
            else:
                reconstruction, _ = model(images)
                loss = nn.functional.binary_cross_entropy(reconstruction, images, reduction="sum") / images.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        print(f"   epoch {epoch:>2}: loss {total_loss / batch_count:.2f}")


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--quick", action="store_true", help="2 epochs instead of 10")
    parsed_arguments = argument_parser.parse_args()

    device = select_best_available_device()
    torch.manual_seed(42)
    epochs = 2 if parsed_arguments.quick else 10

    training_dataset, test_dataset = load_mnist_datasets(flatten_images_to_vectors=True)
    training_loader = DataLoader(training_dataset, batch_size=128, shuffle=True)

    print("1. Plain autoencoder: compress each digit to 2 numbers, reconstruct")
    autoencoder = Autoencoder()
    train_model(autoencoder, False, training_loader, device, epochs)

    sample_image = test_dataset[0][0].to(device)
    autoencoder.eval()
    with torch.no_grad():
        reconstruction, latent_code = autoencoder(sample_image[None])
    print(f"   a '{test_dataset[0][1]}' compressed to just ({latent_code[0, 0]:+.2f}, {latent_code[0, 1]:+.2f}) and back:")
    print("   original:");     print(render_digit_ascii(sample_image.cpu()))
    print("   reconstructed:"); print(render_digit_ascii(reconstruction[0].cpu()))
    print("   784 pixels rebuilt from 2 numbers - digits live on a 2-D surface, not in 784-D chaos.")

    print("\n2. Variational autoencoder: the same, made GENERATIVE")
    vae = VariationalAutoencoder()
    train_model(vae, True, training_loader, device, epochs)

    print("\n   Decoding NEW digits from random points in the latent space")
    print("   (these codes were never produced by any real image):")
    vae.eval()
    with torch.no_grad():
        for point in [(-1.5, -1.5), (0.0, 0.0), (1.5, 1.5)]:
            latent_point = torch.tensor([point], device=device)
            generated = vae.decoder(latent_point)
            print(f"   latent point {point}:")
            print(render_digit_ascii(generated[0].cpu()))
    print("   Each is a coherent digit the model invented - because the VAE's latent")
    print("   space is smooth and full: sample anywhere, decode something real.")


if __name__ == "__main__":
    main()
