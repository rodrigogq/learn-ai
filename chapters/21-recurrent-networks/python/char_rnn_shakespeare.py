"""Chapter 21 - a character-level language model with a recurrent network.

The model reads Shakespeare one character at a time, carrying a HIDDEN STATE
(its memory) forward, and at every step predicts the next character. After a
few minutes of training it writes text with recognizable structure: speaker
names in capitals, line breaks, Elizabethan-shaped words.

The RNN cell is implemented by hand (one tanh of two weighted sums) so there
is nothing hidden; the training loop unrolls it through time and lets
Chapter 8's autograd backpropagate through the unrolled graph ("backprop
through time" is just backprop).

Run from the repository root:
    .venv/bin/python chapters/21-recurrent-networks/python/char_rnn_shakespeare.py --quick
    .venv/bin/python chapters/21-recurrent-networks/python/char_rnn_shakespeare.py
"""

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from common.data import load_tiny_shakespeare  # noqa: E402
from common.device import select_best_available_device  # noqa: E402

SEQUENCE_LENGTH = 128
HIDDEN_SIZE = 256
EMBEDDING_SIZE = 64


class HandmadeRNNCell(nn.Module):
    """One step of a vanilla RNN: new_state = tanh(W*input + U*old_state + b).

    Two weighted sums and a tanh - Chapter 7's neuron, where one input is the
    network's own previous output. That self-connection IS the memory.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_transform = nn.Linear(input_size, hidden_size, bias=False)
        self.state_transform = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_vector, previous_state):
        return torch.tanh(self.input_transform(input_vector) + self.state_transform(previous_state))


class CharRNN(nn.Module):
    """Embedding -> handmade RNN unrolled over time -> next-character logits."""

    def __init__(self, vocabulary_size):
        super().__init__()
        self.character_embedding = nn.Embedding(vocabulary_size, EMBEDDING_SIZE)
        self.rnn_cell = HandmadeRNNCell(EMBEDDING_SIZE, HIDDEN_SIZE)
        self.next_character_head = nn.Linear(HIDDEN_SIZE, vocabulary_size)

    def forward(self, character_ids, initial_state=None):
        """Run the recurrence over a whole sequence.

        Arguments:
            character_ids: (batch, time) integer character ids.
            initial_state: (batch, HIDDEN_SIZE) or None for zeros.

        Returns (logits, final_state): logits (batch, time, vocabulary) - a
        next-character prediction at EVERY position (each position is a
        training example; that efficiency is why language models train fast).
        """
        batch_size, time_steps = character_ids.shape
        hidden_state = initial_state
        if hidden_state is None:
            hidden_state = torch.zeros(batch_size, HIDDEN_SIZE, device=character_ids.device)

        embedded = self.character_embedding(character_ids)
        logits_per_step = []
        # The explicit time loop is the unrolling: the same cell (same
        # weights!) applied at every step, states chaining forward.
        for time_index in range(time_steps):
            hidden_state = self.rnn_cell(embedded[:, time_index], hidden_state)
            logits_per_step.append(self.next_character_head(hidden_state))
        return torch.stack(logits_per_step, dim=1), hidden_state


@torch.no_grad()
def sample_text(model, index_to_character, character_to_index, device,
                prime_text="ROMEO:", length=400, temperature=0.8):
    """Generate text one character at a time, feeding each choice back in.

    Arguments:
        prime_text: the prompt; the state 'warms up' on it first.
        length: characters to generate.
        temperature: divides the logits before softmax - below 1 the model
            plays safe (repetitive), above 1 it gambles (chaotic). 0.8 is a
            pleasant middle.
    """
    model.eval()
    hidden_state = torch.zeros(1, HIDDEN_SIZE, device=device)
    for character in prime_text[:-1]:
        ids = torch.tensor([[character_to_index[character]]], device=device)
        _, hidden_state = model(ids, hidden_state)

    current_id = torch.tensor([[character_to_index[prime_text[-1]]]], device=device)
    generated_characters = [prime_text]
    for _ in range(length):
        logits, hidden_state = model(current_id, hidden_state)
        probabilities = torch.softmax(logits[0, -1] / temperature, dim=0)
        next_id = torch.multinomial(probabilities, 1)
        generated_characters.append(index_to_character[int(next_id)])
        current_id = next_id.view(1, 1)
    model.train()
    return "".join(generated_characters)


def main():
    argument_parser = argparse.ArgumentParser(description=__doc__)
    argument_parser.add_argument("--quick", action="store_true", help="600 steps instead of 3000")
    parsed_arguments = argument_parser.parse_args()

    device = select_best_available_device()
    torch.manual_seed(42)
    total_steps = 600 if parsed_arguments.quick else 3000

    text = load_tiny_shakespeare()
    characters = sorted(set(text))
    character_to_index = {character: index for index, character in enumerate(characters)}
    index_to_character = characters
    corpus = torch.tensor([character_to_index[c] for c in text], dtype=torch.long)
    print(f"Corpus: {len(text):,} characters, vocabulary of {len(characters)} distinct ones")

    model = CharRNN(len(characters)).to(device)
    print(f"CharRNN: {sum(p.numel() for p in model.parameters()):,} parameters")
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_function = nn.CrossEntropyLoss()
    batch_generator = torch.Generator().manual_seed(42)

    print(f"\nTraining for {total_steps} steps...")
    print("   step    loss    (2.9 = random guessing over 65 chars; ~1.4 = decent char model)")
    for step_number in range(1, total_steps + 1):
        # Random windows of the corpus; inputs are the window, targets are the
        # window shifted by one - "predict the next character" at every position.
        starts = torch.randint(0, len(corpus) - SEQUENCE_LENGTH - 1, (64,), generator=batch_generator)
        input_ids = torch.stack([corpus[s:s + SEQUENCE_LENGTH] for s in starts]).to(device)
        target_ids = torch.stack([corpus[s + 1:s + SEQUENCE_LENGTH + 1] for s in starts]).to(device)

        logits, _ = model(input_ids)
        loss = loss_function(logits.reshape(-1, len(characters)), target_ids.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping: through 128 time steps, gradients can snowball
        # (the exploding cousin of vanishing); capping their norm is the
        # standard one-line cure for recurrent training.
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step_number in (1, 100, 500, 1000, 2000, total_steps):
            print(f"  {step_number:>5}   {loss.item():.4f}")

    print("\nSampled text (prime 'ROMEO:', temperature 0.8):")
    print("-" * 60)
    print(sample_text(model, index_to_character, character_to_index, device))
    print("-" * 60)


if __name__ == "__main__":
    main()
