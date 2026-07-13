"""Chapter 28 - a diffusion model on MNIST, from scratch.

Diffusion generates by learning to REVERSE gradual noising. Training: take a
real image, add a known amount of random noise, and teach a network to predict
the noise that was added. Generation: start from pure noise and repeatedly
subtract the network's predicted noise, a little at a time, until an image
emerges. That is DDPM (denoising diffusion probabilistic models), the method
behind modern image generators.

The denoiser is a small time-conditioned U-Net (Chapter 16), told how noisy the
input is via a timestep embedding. The script prints ASCII samples so you can
watch digits crystallize out of static.

Run from the repository root:
    .venv/bin/python chapters/28-diffusion-models/python/train_diffusion_mnist.py --quick
    .venv/bin/python chapters/28-diffusion-models/python/train_diffusion_mnist.py
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

DIFFUSION_STEPS = 200      # how many small noising steps between clean and pure noise


def build_noise_schedule(device):
    """Precompute the noise schedule: how much signal survives at each step.

    Returns alpha_bar[t] = fraction of the ORIGINAL image still present after t
    noising steps (1.0 at t=0, ~0 at the end). The forward noising has a
    closed form: noisy = sqrt(alpha_bar)*image + sqrt(1-alpha_bar)*noise, so we
    can jump to any noise level in one step - no need to simulate the chain.
    """
    betas = torch.linspace(1e-4, 0.02, DIFFUSION_STEPS, device=device)  # per-step noise amount
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)   # alpha_bar


class TimeConditionedUNet(nn.Module):
    """A small U-Net that also receives the timestep, so one network can denoise
    at every noise level (it must behave differently for 'slightly noisy' vs
    'almost pure noise')."""

    def __init__(self):
        super().__init__()
        self.time_embedding = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))

        self.encoder_1 = self._block(1, 32)
        self.encoder_2 = self._block(32, 64)
        self.bottleneck = self._block(64, 128)
        self.downsample = nn.MaxPool2d(2)
        self.upsample_2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder_2 = self._block(128, 64)
        self.upsample_1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decoder_1 = self._block(64, 32)
        self.predict_noise = nn.Conv2d(32, 1, 1)
        # Project the time embedding to each stage's channel count so it can be
        # added into the feature maps.
        self.time_to_bottleneck = nn.Linear(64, 128)

    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.GroupNorm(8, out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.GroupNorm(8, out_channels), nn.SiLU(),
        )

    def forward(self, noisy_images, timesteps):
        # Normalize the timestep to 0..1 and embed it.
        time_features = self.time_embedding((timesteps.float() / DIFFUSION_STEPS)[:, None])

        encoded_1 = self.encoder_1(noisy_images)               # 28
        encoded_2 = self.encoder_2(self.downsample(encoded_1))  # 14
        bottleneck = self.bottleneck(self.downsample(encoded_2))  # 7
        # Inject time information at the bottleneck (broadcast over space).
        bottleneck = bottleneck + self.time_to_bottleneck(time_features)[:, :, None, None]

        decoded_2 = self.decoder_2(torch.cat([self.upsample_2(bottleneck), encoded_2], dim=1))
        decoded_1 = self.decoder_1(torch.cat([self.upsample_1(decoded_2), encoded_1], dim=1))
        return self.predict_noise(decoded_1)


@torch.no_grad()
def sample_image(model, alpha_bar, device):
    """Generate one image by reversing the diffusion, step by step.

    Start from pure noise; at each step predict the noise, remove a slice of
    it, add back a little fresh noise (except at the very end), and continue.
    After DIFFUSION_STEPS reverse steps, a digit remains.
    """
    model.eval()
    image = torch.randn(1, 1, 28, 28, device=device)
    betas = 1.0 - (alpha_bar / torch.cat([torch.ones(1, device=device), alpha_bar[:-1]]))
    for step in reversed(range(DIFFUSION_STEPS)):
        timestep = torch.tensor([step], device=device)
        predicted_noise = model(image, timestep)
        alpha = 1.0 - betas[step]
        # The DDPM reverse-step formula: subtract the scaled prediction, then
        # renormalize; the derivation is standard and given in the chapter refs.
        image = (image - betas[step] / torch.sqrt(1 - alpha_bar[step]) * predicted_noise) / torch.sqrt(alpha)
        if step > 0:
            image = image + torch.sqrt(betas[step]) * torch.randn_like(image)
    model.train()
    return image[0, 0].clamp(0, 1)


def render_digit_ascii(pixels):
    grid = pixels.detach().cpu().reshape(28, 28)
    characters = " .:-=+*#%@"
    lines = []
    for row in range(0, 28, 2):
        line = "".join(characters[min(int(max(grid[row, column].item(), 0) * len(characters)), len(characters) - 1)]
                        for column in range(0, 28, 2))
        lines.append("   " + line)
    return "\n".join(lines)


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--quick", action="store_true", help="1 epoch instead of 6")
    parsed_arguments = argument_parser.parse_args()

    device = select_best_available_device()
    torch.manual_seed(42)
    epochs = 1 if parsed_arguments.quick else 6

    training_dataset, _ = load_mnist_datasets()
    training_loader = DataLoader(training_dataset, batch_size=128, shuffle=True)
    alpha_bar = build_noise_schedule(device)

    model = TimeConditionedUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    print("Training the denoiser: add known noise, predict it back.\n")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batch_count = 0
        for real_images, _ in training_loader:
            real_images = real_images.to(device)
            batch_size = real_images.shape[0]

            # Pick a random noise level per image and jump straight to it
            # (closed form), then ask the network to recover the added noise.
            timesteps = torch.randint(0, DIFFUSION_STEPS, (batch_size,), device=device)
            noise = torch.randn_like(real_images)
            signal_scale = torch.sqrt(alpha_bar[timesteps])[:, None, None, None]
            noise_scale = torch.sqrt(1 - alpha_bar[timesteps])[:, None, None, None]
            noisy_images = signal_scale * real_images + noise_scale * noise

            predicted_noise = model(noisy_images, timesteps)
            loss = nn.functional.mse_loss(predicted_noise, noise)   # just predict the noise!
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        print(f"epoch {epoch}/{epochs}  noise-prediction loss {total_loss / batch_count:.4f}")
        print("  a digit sampled from pure noise:")
        print(render_digit_ascii(sample_image(model, alpha_bar, device)))
        print()

    print("Trained by the simplest objective in this whole course - 'predict the noise,")
    print("mean squared error' - yet it generates digits by turning static into structure,")
    print("one small denoising step at a time. This is how modern image generators work.")


if __name__ == "__main__":
    main()
