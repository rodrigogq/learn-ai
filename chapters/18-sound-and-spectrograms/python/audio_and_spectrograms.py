"""Chapter 18 - sound from zero: waveforms, the Fourier transform, spectrograms,
and a CNN that classifies sounds by looking at them.

Four demonstrations:
  1. sound is a vector: synthesize tones, chords, and chirps as plain arrays,
  2. the discrete Fourier transform, implemented naively from its definition,
     finds the exact notes inside a chord (verified against numpy's FFT),
  3. the spectrogram: many small FFTs over time turn sound into an image,
  4. a small CNN classifies five sound types from their spectrograms - the
     bridge that turns all of Part III's vision machinery into audio machinery.

Run from the repository root:
    .venv/bin/python chapters/18-sound-and-spectrograms/python/audio_and_spectrograms.py --quick
    .venv/bin/python chapters/18-sound-and-spectrograms/python/audio_and_spectrograms.py
"""

import argparse
import sys
from pathlib import Path

import numpy
import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from common.device import select_best_available_device  # noqa: E402

SAMPLE_RATE = 8000          # samples per second - CD quality is 44100; 8000 keeps arrays small
CLIP_SECONDS = 0.5
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
SOUND_CLASS_NAMES = ["pure tone", "chord", "rising chirp", "falling chirp", "noise"]


def synthesize_sound(sound_class, random_generator):
    """Build one clip of the requested class as a plain float array.

    Arguments:
        sound_class: index into SOUND_CLASS_NAMES.
        random_generator: numpy Generator for random pitches and noise.

    A sine wave at frequency f is  sin(2*pi*f*t): the fundamental atom of
    sound. Chords add several; chirps sweep the frequency over time.
    """
    time_axis = numpy.arange(CLIP_SAMPLES) / SAMPLE_RATE
    base_frequency = random_generator.uniform(220.0, 880.0)

    if sound_class == 0:      # pure tone
        waveform = numpy.sin(2 * numpy.pi * base_frequency * time_axis)
    elif sound_class == 1:    # chord: root + major third + fifth (ratios 1 : 1.26 : 1.5)
        waveform = (numpy.sin(2 * numpy.pi * base_frequency * time_axis)
                    + numpy.sin(2 * numpy.pi * base_frequency * 1.26 * time_axis)
                    + numpy.sin(2 * numpy.pi * base_frequency * 1.5 * time_axis)) / 3.0
    elif sound_class in (2, 3):  # chirps: frequency slides up or down one octave
        end_frequency = base_frequency * (2.0 if sound_class == 2 else 0.5)
        # The instantaneous frequency moves linearly, so the phase is the
        # integral of it - the standard linear-chirp formula.
        swept_phase = 2 * numpy.pi * (base_frequency * time_axis
                                      + (end_frequency - base_frequency) * time_axis ** 2 / (2 * CLIP_SECONDS))
        waveform = numpy.sin(swept_phase)
    else:                     # noise
        waveform = random_generator.normal(0.0, 0.5, CLIP_SAMPLES)

    return waveform + random_generator.normal(0.0, 0.02, CLIP_SAMPLES)


def naive_discrete_fourier_transform(waveform):
    """The DFT exactly as defined - O(N^2), no tricks, for understanding.

    Arguments:
        waveform: array of N samples.

    For each candidate frequency k (in cycles per window), correlate the
    signal with a cosine and a sine of that frequency (two dot products -
    Chapter 2's 'how aligned are these?' operation). Strong correlation
    means the frequency is present. Returns the magnitude per frequency.
    """
    sample_count = len(waveform)
    sample_indices = numpy.arange(sample_count)
    magnitudes = numpy.zeros(sample_count // 2)
    for frequency_index in range(sample_count // 2):
        angles = 2 * numpy.pi * frequency_index * sample_indices / sample_count
        cosine_correlation = (waveform * numpy.cos(angles)).sum()
        sine_correlation = (waveform * numpy.sin(angles)).sum()
        magnitudes[frequency_index] = numpy.sqrt(cosine_correlation ** 2 + sine_correlation ** 2)
    return magnitudes


def compute_spectrogram(waveform, window_size=256, hop_size=128):
    """Sound as an image: FFT magnitudes of overlapping windows, stacked.

    Arguments:
        waveform: the full clip.
        window_size: samples per FFT window (256 samples = 32 ms here).
        hop_size: samples between window starts (50% overlap).

    Returns an array (frequency_bins, time_frames) of log-magnitudes - the
    standard 'sound picture' with time on x and pitch on y.
    """
    window_function = numpy.hanning(window_size)  # tapers window edges to avoid artificial clicks
    frames = []
    for start in range(0, len(waveform) - window_size + 1, hop_size):
        windowed = waveform[start:start + window_size] * window_function
        magnitudes = numpy.abs(numpy.fft.rfft(windowed))
        frames.append(numpy.log(magnitudes + 1e-6))
    return numpy.stack(frames, axis=1)


def demonstrate_fourier_transform():
    """Demo 2: find the notes inside an A-major chord, from the definition."""
    print("2. The Fourier transform finds the notes inside a chord")
    time_axis = numpy.arange(1024) / SAMPLE_RATE
    chord = (numpy.sin(2 * numpy.pi * 440.0 * time_axis)
             + numpy.sin(2 * numpy.pi * 554.4 * time_axis)
             + numpy.sin(2 * numpy.pi * 659.3 * time_axis))

    magnitudes = naive_discrete_fourier_transform(chord)
    frequency_resolution = SAMPLE_RATE / 1024
    peak_indices = [int(i) for i in numpy.argsort(magnitudes)[-3:]]
    peak_frequencies = sorted(index * frequency_resolution for index in peak_indices)
    print(f"   input: A major chord, notes at 440.0, 554.4, 659.3 Hz")
    print(f"   naive DFT's three strongest frequencies: "
          + ", ".join(f"{frequency:.1f} Hz" for frequency in peak_frequencies))

    fft_magnitudes = numpy.abs(numpy.fft.rfft(chord))[:512]
    agreement = numpy.allclose(magnitudes, fft_magnitudes, atol=1e-6)
    print(f"   numpy's FFT computes the same numbers: {agreement} "
          f"(the FFT is the same math, reorganized from O(N^2) to O(N log N))")


class SpectrogramCNN(nn.Module):
    """A small CNN over spectrogram 'images' - Part III machinery, reused as-is."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, len(SOUND_CLASS_NAMES)),
        )

    def forward(self, spectrogram_batch):
        return self.network(spectrogram_batch)


def build_spectrogram_batch(batch_size, random_generator):
    """Synthesize a batch of labeled sounds and return their spectrograms."""
    labels = random_generator.integers(0, len(SOUND_CLASS_NAMES), batch_size)
    spectrograms = []
    for label in labels:
        spectrogram = compute_spectrogram(synthesize_sound(int(label), random_generator))
        spectrograms.append(spectrogram)
    batch = torch.tensor(numpy.stack(spectrograms), dtype=torch.float32)[:, None]
    # Per-batch standardization (Chapter 5's rule): log-magnitudes span a wide
    # range; the network trains far better on mean-0 inputs.
    batch = (batch - batch.mean()) / (batch.std() + 1e-6)
    return batch, torch.tensor(labels, dtype=torch.long)


def train_sound_classifier(device, total_steps):
    """Demo 4: five sound classes from spectrograms."""
    print()
    print(f"4. Training a CNN on spectrograms ({total_steps} steps, fresh sounds each step)")
    random_generator = numpy.random.default_rng(42)
    model = SpectrogramCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()

    for step_number in range(1, total_steps + 1):
        spectrograms, labels = build_spectrogram_batch(32, random_generator)
        loss = loss_function(model(spectrograms.to(device)), labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step_number in (1, total_steps // 2, total_steps):
            evaluation_generator = numpy.random.default_rng(999)
            spectrograms, labels = build_spectrogram_batch(200, evaluation_generator)
            model.eval()
            with torch.no_grad():
                predictions = model(spectrograms.to(device)).argmax(dim=1).cpu()
            model.train()
            accuracy = (predictions == labels).float().mean().item()
            print(f"   step {step_number:>4}: loss {loss.item():.4f}, accuracy on fresh sounds {accuracy:.1%}")

    print("   Rising vs falling chirps have IDENTICAL frequency content overall -")
    print("   only the spectrogram's time axis separates them. Sound became vision.")


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--quick", action="store_true", help="150 training steps instead of 500")
    parsed_arguments = argument_parser.parse_args()

    print("1. Sound is a vector")
    random_generator = numpy.random.default_rng(42)
    tone = synthesize_sound(0, random_generator)
    print(f"   half a second at {SAMPLE_RATE} samples/second = {len(tone)} numbers")
    print(f"   first five samples of a tone: {numpy.round(tone[:5], 3).tolist()}")

    print()
    demonstrate_fourier_transform()

    print()
    print("3. The spectrogram: many small FFTs, stacked into an image")
    spectrogram = compute_spectrogram(synthesize_sound(2, random_generator))
    print(f"   one clip -> spectrogram of shape {spectrogram.shape} (frequency bins x time frames)")
    print("   a rising chirp paints a diagonal stripe in it - see the chapter figure.")

    device = select_best_available_device()
    train_sound_classifier(device, 150 if parsed_arguments.quick else 500)


if __name__ == "__main__":
    main()
