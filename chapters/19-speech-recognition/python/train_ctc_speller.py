"""Chapter 19 - speech-to-text in miniature: CTC learns to spell what it hears.

The core problem of speech recognition is ALIGNMENT: audio has hundreds of
frames, text has a few letters, and nobody tells you which frames belong to
which letter. CTC (connectionist temporal classification) solves it: let the
network output one symbol per FRAME (letters + a special 'blank'), then
collapse repeats and remove blanks to get the text - and train by summing
probability over every alignment that collapses to the truth (torch's
nn.CTCLoss does that sum).

The toy language: words of 2-5 letters from {A,B,C,D,E}; "speaking" a letter
means playing its tone (A=300Hz ... E=1100Hz) for a RANDOM duration. The
model hears a spectrogram and must output the letter sequence - never told
where letters start or end.

Run from the repository root:
    .venv/bin/python chapters/19-speech-recognition/python/train_ctc_speller.py --quick
    .venv/bin/python chapters/19-speech-recognition/python/train_ctc_speller.py
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

SAMPLE_RATE = 8000
WINDOW_SIZE = 128
HOP_SIZE = 64
FREQUENCY_BINS = WINDOW_SIZE // 2 + 1

ALPHABET = "ABCDE"
LETTER_FREQUENCIES = {"A": 300.0, "B": 500.0, "C": 700.0, "D": 900.0, "E": 1100.0}
BLANK_INDEX = len(ALPHABET)          # CTC's extra symbol lives at the last index


def speak_word(word, random_generator):
    """Synthesize a word: each letter's tone, held for a random duration.

    Arguments:
        word: string over the ALPHABET.
        random_generator: numpy Generator (durations and noise vary per call -
            the same word never sounds exactly the same twice, like speech).

    Returns the waveform as a float array.
    """
    pieces = []
    for letter in word:
        duration_samples = int(random_generator.uniform(0.08, 0.2) * SAMPLE_RATE)
        time_axis = numpy.arange(duration_samples) / SAMPLE_RATE
        pieces.append(numpy.sin(2 * numpy.pi * LETTER_FREQUENCIES[letter] * time_axis))
    waveform = numpy.concatenate(pieces)
    return waveform + random_generator.normal(0.0, 0.05, len(waveform))


def compute_spectrogram(waveform):
    """Chapter 18's STFT, condensed: log-magnitude frames of windowed FFTs."""
    window_function = numpy.hanning(WINDOW_SIZE)
    frames = []
    for start in range(0, len(waveform) - WINDOW_SIZE + 1, HOP_SIZE):
        windowed = waveform[start:start + WINDOW_SIZE] * window_function
        frames.append(numpy.log(numpy.abs(numpy.fft.rfft(windowed)) + 1e-6))
    return numpy.stack(frames, axis=1)  # (FREQUENCY_BINS, frame_count)


def build_word_batch(batch_size, random_generator):
    """Random words, spoken and spectrogram-ed, padded into one batch.

    Returns (spectrograms, frame_counts, targets, target_lengths, words):
    everything nn.CTCLoss needs, plus the words for humans.
    """
    words = []
    spectrograms = []
    for _ in range(batch_size):
        word_length = int(random_generator.integers(2, 6))
        word = "".join(ALPHABET[int(random_generator.integers(len(ALPHABET)))] for _ in range(word_length))
        words.append(word)
        spectrograms.append(compute_spectrogram(speak_word(word, random_generator)))

    frame_counts = torch.tensor([s.shape[1] for s in spectrograms])
    longest = int(frame_counts.max())
    batch = torch.zeros(batch_size, FREQUENCY_BINS, longest)
    for i, spectrogram in enumerate(spectrograms):
        batch[i, :, :spectrogram.shape[1]] = torch.tensor(spectrogram, dtype=torch.float32)
    batch = (batch - batch.mean()) / (batch.std() + 1e-6)

    targets = torch.tensor([ALPHABET.index(letter) for word in words for letter in word])
    target_lengths = torch.tensor([len(word) for word in words])
    return batch, frame_counts, targets, target_lengths, words


class FramewiseSpeller(nn.Module):
    """1D convolutions over time, then a per-frame letter classifier.

    Each output frame gets logits over {A..E, blank}. No recurrence needed
    here: a few conv layers see enough temporal context for tones. (Real
    speech models put a sequence model here - Chapters 21-22 build those.)
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(FREQUENCY_BINS, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(),
        )
        self.per_frame_classifier = nn.Conv1d(64, len(ALPHABET) + 1, kernel_size=1)

    def forward(self, spectrogram_batch):
        return self.per_frame_classifier(self.encoder(spectrogram_batch))  # (batch, 6, frames)


def greedy_ctc_decode(framewise_logits):
    """The CTC collapse rule: argmax per frame, merge repeats, drop blanks.

    Arguments:
        framewise_logits: (classes, frames) for ONE example.

    Returns the decoded string. This is 'greedy' decoding - taking the best
    symbol per frame independently; beam search would track alternatives.
    """
    best_indices = framewise_logits.argmax(dim=0).tolist()
    decoded_letters = []
    previous_index = None
    for index in best_indices:
        # Merge repeats FIRST, then drop blanks - order matters: the blank is
        # what lets CTC spell double letters like "AAB" -> A,blank,A,B.
        if index != previous_index and index != BLANK_INDEX:
            decoded_letters.append(ALPHABET[index])
        previous_index = index
    return "".join(decoded_letters)


def evaluate(model, device, sample_count=100):
    """Exact-match accuracy on freshly spoken words."""
    evaluation_generator = numpy.random.default_rng(999)
    batch, frame_counts, _, _, words = build_word_batch(sample_count, evaluation_generator)
    model.eval()
    with torch.no_grad():
        logits = model(batch.to(device)).cpu()
    model.train()
    correct = sum(greedy_ctc_decode(logits[i, :, :frame_counts[i]]) == words[i]
                  for i in range(sample_count))
    return correct / sample_count


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--quick", action="store_true", help="300 steps instead of 1000")
    parsed_arguments = argument_parser.parse_args()

    device = select_best_available_device()
    torch.manual_seed(42)
    random_generator = numpy.random.default_rng(42)
    total_steps = 300 if parsed_arguments.quick else 1000

    model = FramewiseSpeller().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # zero_infinity guards against the rare degenerate batch where a word is
    # longer than its frame count would allow.
    ctc_loss = nn.CTCLoss(blank=BLANK_INDEX, zero_infinity=True)

    print(f"Training the CTC speller for {total_steps} steps...")
    print("   step    CTC loss   exact-match accuracy on 100 fresh words")
    for step_number in range(1, total_steps + 1):
        batch, frame_counts, targets, target_lengths, _ = build_word_batch(16, random_generator)
        framewise_logits = model(batch.to(device))
        # CTCLoss wants (frames, batch, classes) log-probabilities.
        log_probabilities = framewise_logits.permute(2, 0, 1).log_softmax(dim=2)
        # Apple's MPS backend has no CTC loss kernel (a real-world lesson:
        # backends have gaps). Moving the log-probabilities to the CPU for
        # the loss works because autograd routes gradients back across the
        # device copy; the conv encoder still runs on the GPU.
        if device.type == "mps":
            log_probabilities = log_probabilities.cpu()
            loss = ctc_loss(log_probabilities, targets, frame_counts, target_lengths)
        else:
            loss = ctc_loss(log_probabilities, targets.to(device),
                            frame_counts.to(device), target_lengths.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_number in (1, 100, total_steps // 2, total_steps):
            accuracy = evaluate(model, device)
            print(f"  {step_number:>5}   {loss.item():>8.4f}   {accuracy:.1%}")

    print()
    print("Five fresh words, spoken and transcribed:")
    demo_generator = numpy.random.default_rng(7)
    batch, frame_counts, _, _, words = build_word_batch(5, demo_generator)
    model.eval()
    with torch.no_grad():
        logits = model(batch.to(device)).cpu()
    for i in range(5):
        frame_logits = logits[i, :, :frame_counts[i]]
        raw_frame_symbols = "".join(
            "-" if int(index) == BLANK_INDEX else ALPHABET[int(index)]
            for index in frame_logits.argmax(dim=0)
        )
        print(f"  spoken '{words[i]}'  ->  frames: {raw_frame_symbols}")
        print(f"           -> collapsed: '{greedy_ctc_decode(frame_logits)}'")


if __name__ == "__main__":
    main()
