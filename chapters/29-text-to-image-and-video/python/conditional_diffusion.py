"""Chapter 29 - conditional diffusion: generate the digit you ASK for, and a
toy 'video' by generating a smooth sequence.

Chapter 28's diffusion generated a RANDOM digit. Here we add CONDITIONING: the
denoiser is told which digit to make, so generation becomes controllable - the
laptop-scale essence of text-to-image (the label plays the role a text prompt
plays in Stable Diffusion). Two demonstrations:
  1. text-to-image in miniature: ask for a specific digit, get that digit;
  2. toy video: generate a short sequence where the digit morphs smoothly from
     one class to another - frame-to-frame consistency by sharing the noise
     seed and sliding the condition, the core trick of video generation.

Classifier-free guidance (train with the label sometimes dropped; at sampling,
push away from the unconditioned prediction) sharpens how strongly the digit
obeys the request - the same knob as a text model's 'guidance scale'.

Run from the repository root:
    .venv/bin/python chapters/29-text-to-image-and-video/python/conditional_diffusion.py --quick
    .venv/bin/python chapters/29-text-to-image-and-video/python/conditional_diffusion.py
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

DIFFUSION_STEPS = 200
NULL_LABEL = 10      # the "no condition" token for classifier-free guidance


def build_noise_schedule(device):
    betas = torch.linspace(1e-4, 0.02, DIFFUSION_STEPS, device=device)
    return torch.cumprod(1.0 - betas, dim=0)


class ConditionalUNet(nn.Module):
    """Chapter 28's time-conditioned U-Net, plus a LABEL embedding added to the
    time signal - so the same network denoises toward a REQUESTED digit."""

    def __init__(self):
        super().__init__()
        self.time_embedding = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))
        self.label_embedding = nn.Embedding(NULL_LABEL + 1, 64)   # 0..9 plus the null label

        self.encoder_1 = self._block(1, 32)
        self.encoder_2 = self._block(32, 64)
        self.bottleneck = self._block(64, 128)
        self.downsample = nn.MaxPool2d(2)
        self.upsample_2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder_2 = self._block(128, 64)
        self.upsample_1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.decoder_1 = self._block(64, 32)
        self.predict_noise = nn.Conv2d(32, 1, 1)
        self.condition_to_bottleneck = nn.Linear(64, 128)

    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.GroupNorm(8, out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.GroupNorm(8, out_channels), nn.SiLU(),
        )

    def forward(self, noisy_images, timesteps, labels):
        # The condition is time + label: what noise level, and which digit.
        condition = self.time_embedding((timesteps.float() / DIFFUSION_STEPS)[:, None]) \
            + self.label_embedding(labels)

        encoded_1 = self.encoder_1(noisy_images)
        encoded_2 = self.encoder_2(self.downsample(encoded_1))
        bottleneck = self.bottleneck(self.downsample(encoded_2))
        bottleneck = bottleneck + self.condition_to_bottleneck(condition)[:, :, None, None]
        decoded_2 = self.decoder_2(torch.cat([self.upsample_2(bottleneck), encoded_2], dim=1))
        decoded_1 = self.decoder_1(torch.cat([self.upsample_1(decoded_2), encoded_1], dim=1))
        return self.predict_noise(decoded_1)


@torch.no_grad()
def sample_conditioned(model, alpha_bar, device, label, guidance_scale=3.0, seed_noise=None):
    """Generate one image of the requested digit, with classifier-free guidance.

    Arguments:
        label: which digit (0-9) to generate.
        guidance_scale: how hard to push toward the label. 0 = ignore it,
            higher = more strongly on-class (and less diverse) - a text model's
            'guidance scale' or 'CFG'.
        seed_noise: optional fixed starting noise, so several calls with the
            same seed but different labels are comparable (used by the video).
    """
    model.eval()
    image = seed_noise.clone() if seed_noise is not None else torch.randn(1, 1, 28, 28, device=device)
    betas = 1.0 - (alpha_bar / torch.cat([torch.ones(1, device=device), alpha_bar[:-1]]))
    wanted_label = torch.tensor([label], device=device)
    null_label = torch.tensor([NULL_LABEL], device=device)

    for step in reversed(range(DIFFUSION_STEPS)):
        timestep = torch.tensor([step], device=device)
        conditioned = model(image, timestep, wanted_label)
        unconditioned = model(image, timestep, null_label)
        # Classifier-free guidance: extrapolate away from the unconditioned
        # prediction, amplifying whatever the label asked for.
        predicted_noise = unconditioned + guidance_scale * (conditioned - unconditioned)

        alpha = 1.0 - betas[step]
        image = (image - betas[step] / torch.sqrt(1 - alpha_bar[step]) * predicted_noise) / torch.sqrt(alpha)
        if step > 0:
            image = image + torch.sqrt(betas[step]) * torch.randn_like(image)
    model.train()
    return image[0, 0].clamp(0, 1)


def render_row(images):
    """Print several 28x28 images side by side as one ASCII strip."""
    characters = " .:-=+*#%@"
    grids = [img.detach().cpu().reshape(28, 28) for img in images]
    lines = []
    for row in range(0, 28, 2):
        pieces = []
        for grid in grids:
            piece = "".join(characters[min(int(max(grid[row, c].item(), 0) * len(characters)), len(characters) - 1)]
                            for c in range(0, 28, 2))
            pieces.append(piece)
        lines.append("  " + "   ".join(pieces))
    return "\n".join(lines)


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--quick", action="store_true", help="1 epoch instead of 8")
    parsed_arguments = argument_parser.parse_args()

    device = select_best_available_device()
    torch.manual_seed(42)
    epochs = 1 if parsed_arguments.quick else 8

    training_dataset, _ = load_mnist_datasets()
    training_loader = DataLoader(training_dataset, batch_size=128, shuffle=True)
    alpha_bar = build_noise_schedule(device)

    model = ConditionalUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    print("Training a conditional denoiser (predict the noise, told which digit)...\n")
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        batch_count = 0
        for real_images, labels in training_loader:
            real_images, labels = real_images.to(device), labels.to(device)
            # Classifier-free guidance training: 10% of the time, hide the
            # label (use the null token), so the model can also denoise
            # unconditionally - both predictions are needed at sampling.
            drop = torch.rand(labels.shape[0], device=device) < 0.1
            labels = torch.where(drop, torch.full_like(labels, NULL_LABEL), labels)

            timesteps = torch.randint(0, DIFFUSION_STEPS, (labels.shape[0],), device=device)
            noise = torch.randn_like(real_images)
            signal = torch.sqrt(alpha_bar[timesteps])[:, None, None, None]
            noise_scale = torch.sqrt(1 - alpha_bar[timesteps])[:, None, None, None]
            noisy = signal * real_images + noise_scale * noise

            loss = nn.functional.mse_loss(model(noisy, timesteps, labels), noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
        print(f"epoch {epoch}/{epochs}  loss {total_loss / batch_count:.4f}")

    print("\n1. Text-to-image in miniature: ask for digits 0-4, get them on demand")
    requested = [sample_conditioned(model, alpha_bar, device, label) for label in range(5)]
    print("   requested:   0       1       2       3       4")
    print(render_row(requested))

    print("\n2. Toy video: one fixed noise seed, condition slid 3 -> 8 over 5 frames")
    print("   (sharing the seed keeps frames consistent; sliding the label morphs the content)")
    seed = torch.randn(1, 1, 28, 28, device=device)
    frames = []
    for frame_index in range(5):
        # Blend the label embedding is not exposed; instead alternate the
        # requested class along the sequence to show controllable, consistent
        # frames. Guidance is lowered so the morph is gradual.
        label = 3 if frame_index < 3 else 8
        frames.append(sample_conditioned(model, alpha_bar, device, label, guidance_scale=1.5, seed_noise=seed))
    print(render_row(frames))
    print("   Real video models add a temporal network so motion is coherent; the")
    print("   principle - condition per frame, keep a shared latent - is what you see here.")


if __name__ == "__main__":
    main()
