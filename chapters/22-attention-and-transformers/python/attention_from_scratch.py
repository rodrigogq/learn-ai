"""Chapter 22 - attention from scratch: the worked example, verification, and
a model you can watch deciding where to look.

Three demonstrations:
  1. the chapter's tiny worked example, computed exactly (3 tokens, 2 dims),
     with and without the causal mask - matching the README's numbers,
  2. our implementation agrees with PyTorch's fused attention kernel,
  3. a key-value lookup task: a single attention layer learns to fetch the
     right item, and PRINTING ITS ATTENTION WEIGHTS shows it looking at
     exactly the matching position. Content-based addressing, learned.

Run from the repository root:
    .venv/bin/python chapters/22-attention-and-transformers/python/attention_from_scratch.py
"""

import math
import sys
from pathlib import Path

import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from common.device import select_best_available_device  # noqa: E402


def scaled_dot_product_attention(queries, keys, values, causal_mask=False):
    """The whole attention mechanism - four lines.

    Arguments:
        queries: (tokens, key_size) - what each position is looking for.
        keys:    (tokens, key_size) - what each position offers to be found by.
        values:  (tokens, value_size) - what each position hands over if chosen.
        causal_mask: hide the future (position i may only attend to <= i).

    Returns (output, attention_weights). Each output row is a weighted average
    of ALL value rows, weighted by how well this position's query matches
    every position's key (dot products, softmaxed).
    """
    key_size = queries.shape[-1]
    # Scores: every query dotted with every key. The sqrt scaling keeps the
    # dot products from growing with dimension and saturating the softmax.
    match_scores = queries @ keys.T / math.sqrt(key_size)
    if causal_mask:
        future_positions = torch.triu(torch.ones_like(match_scores, dtype=torch.bool), diagonal=1)
        match_scores = match_scores.masked_fill(future_positions, float("-inf"))
    attention_weights = torch.softmax(match_scores, dim=-1)
    return attention_weights @ values, attention_weights


def demonstrate_worked_example():
    """Demo 1: the README's numbers, computed."""
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    keys = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    values = torch.tensor([[1.0, 0.0], [10.0, 0.0], [0.0, 10.0]])

    print("1. The worked example (3 tokens, 2 dimensions)")
    output, weights = scaled_dot_product_attention(queries, keys, values)
    print("   raw scores (Q K^T / sqrt(2)):")
    for row in (queries @ keys.T / math.sqrt(2)):
        print("     " + "  ".join(f"{value:+.3f}" for value in row))
    print("   attention weights (each row softmaxed, sums to 1):")
    for row in weights:
        print("     " + "  ".join(f"{value:.3f}" for value in row))
    print("   output (weights @ V):")
    for row in output:
        print("     " + "  ".join(f"{value:.3f}" for value in row))

    print("\n   with the causal mask (each position sees only itself and the past):")
    _, causal_weights = scaled_dot_product_attention(queries, keys, values, causal_mask=True)
    for row in causal_weights:
        print("     " + "  ".join(f"{value:.3f}" for value in row))


def verify_against_pytorch():
    """Demo 2: same numbers as torch's built-in fused attention."""
    generator = torch.Generator().manual_seed(42)
    queries = torch.randn(8, 16, generator=generator)
    keys = torch.randn(8, 16, generator=generator)
    values = torch.randn(8, 16, generator=generator)

    our_output, _ = scaled_dot_product_attention(queries, keys, values, causal_mask=True)
    torch_output = nn.functional.scaled_dot_product_attention(
        queries[None], keys[None], values[None], is_causal=True
    )[0]
    largest_difference = (our_output - torch_output).abs().max().item()
    print(f"\n2. Against torch.nn.functional.scaled_dot_product_attention: "
          f"largest difference {largest_difference:.2e} (agreement)")


# ---------------------------------------------------------- the lookup task

KEY_COUNT = 5
VALUE_COUNT = 5
ITEM_TOKENS = KEY_COUNT * VALUE_COUNT      # token id = key*5 + value, a "key:value" card
QUERY_TOKENS = KEY_COUNT                   # token id = ITEM_TOKENS + key
VOCABULARY_SIZE = ITEM_TOKENS + QUERY_TOKENS
ITEMS_PER_SEQUENCE = 4


class OneAttentionLayerModel(nn.Module):
    """Embedding -> ONE attention layer -> classifier. No recurrence, no convs."""

    def __init__(self, embedding_size=32):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCABULARY_SIZE, embedding_size)
        self.query_projection = nn.Linear(embedding_size, embedding_size, bias=False)
        self.key_projection = nn.Linear(embedding_size, embedding_size, bias=False)
        self.value_projection = nn.Linear(embedding_size, embedding_size, bias=False)
        self.answer_head = nn.Linear(embedding_size, VALUE_COUNT)

    def forward(self, token_ids):
        """Returns (logits for the answer, the last position's attention row)."""
        embedded = self.token_embedding(token_ids)                       # (batch, seq, emb)
        queries = self.query_projection(embedded)
        keys = self.key_projection(embedded)
        values = self.value_projection(embedded)

        scores = queries @ keys.transpose(1, 2) / math.sqrt(queries.shape[-1])
        attention_weights = torch.softmax(scores, dim=-1)
        attended = attention_weights @ values
        # Only the final position (the query token) must answer.
        return self.answer_head(attended[:, -1]), attention_weights[:, -1]


def build_lookup_batch(batch_size, generator):
    """Sequences of 4 'key:value cards' plus one query; target = looked-up value.

    Example (as humans would read it):  [k2:v4  k0:v1  k3:v3  k1:v0  query k3]
    -> answer v3. The model must find WHERE the matching card is (different
    every time) and read its value - content-based addressing.
    """
    sequences = torch.zeros(batch_size, ITEMS_PER_SEQUENCE + 1, dtype=torch.long)
    answers = torch.zeros(batch_size, dtype=torch.long)
    for i in range(batch_size):
        shuffled_keys = torch.randperm(KEY_COUNT, generator=generator)[:ITEMS_PER_SEQUENCE]
        card_values = torch.randint(0, VALUE_COUNT, (ITEMS_PER_SEQUENCE,), generator=generator)
        sequences[i, :ITEMS_PER_SEQUENCE] = shuffled_keys * VALUE_COUNT + card_values
        chosen_position = int(torch.randint(0, ITEMS_PER_SEQUENCE, (1,), generator=generator))
        sequences[i, -1] = ITEM_TOKENS + shuffled_keys[chosen_position]
        answers[i] = card_values[chosen_position]
    return sequences, answers


def train_lookup_model(device):
    """Demo 3: train, then show the attention weights doing the looking."""
    print("\n3. The key-value lookup task: one attention layer, watch it learn to look")
    generator = torch.Generator().manual_seed(42)
    model = OneAttentionLayerModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()

    for step_number in range(1, 1501):
        sequences, answers = build_lookup_batch(64, generator)
        logits, _ = model(sequences.to(device))
        loss = loss_function(logits, answers.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step_number in (1, 250, 750, 1500):
            evaluation_generator = torch.Generator().manual_seed(999)
            sequences, answers = build_lookup_batch(500, evaluation_generator)
            model.eval()
            with torch.no_grad():
                logits, _ = model(sequences.to(device))
            model.train()
            accuracy = (logits.argmax(dim=1).cpu() == answers).float().mean().item()
            print(f"   step {step_number:>4}: loss {loss.item():.4f}, accuracy {accuracy:.1%}")

    print("\n   One test sequence, and where the query's attention actually went:")
    demonstration_generator = torch.Generator().manual_seed(7)
    sequences, answers = build_lookup_batch(1, demonstration_generator)
    model.eval()
    with torch.no_grad():
        logits, attention_row = model(sequences.to(device))
    tokens_text = []
    for token_id in sequences[0].tolist():
        if token_id < ITEM_TOKENS:
            tokens_text.append(f"k{token_id // VALUE_COUNT}:v{token_id % VALUE_COUNT}")
        else:
            tokens_text.append(f"QUERY k{token_id - ITEM_TOKENS}")
    for token_text, weight in zip(tokens_text, attention_row[0].tolist()):
        marker = "  <-- it looked HERE" if weight == max(attention_row[0].tolist()) else ""
        print(f"     {token_text:<10} attention {weight:.3f}{marker}")
    print(f"     model answers v{int(logits.argmax())}, truth v{int(answers[0])}")


def main():
    demonstrate_worked_example()
    verify_against_pytorch()
    device = select_best_available_device(print_choice=False)
    train_lookup_model(device)


if __name__ == "__main__":
    main()
