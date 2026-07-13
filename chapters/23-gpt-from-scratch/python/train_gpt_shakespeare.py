"""Chapter 23 - a GPT, from scratch, trained on Shakespeare in minutes.

The full decoder-only transformer, assembled from Chapter 22's parts: token
embeddings + positional embeddings, four transformer blocks (multi-head causal
attention, MLP, residuals, layer norm), and a next-token head. Character-level
here for transparency (Chapter 24 upgrades to BPE tokens and scale).

The script samples from the model BEFORE training and after, so you can see
exactly what the gradients bought.

Run from the repository root:
    .venv/bin/python chapters/23-gpt-from-scratch/python/train_gpt_shakespeare.py --quick
    .venv/bin/python chapters/23-gpt-from-scratch/python/train_gpt_shakespeare.py
"""

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch import nn

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT))

from common.data import load_tiny_shakespeare  # noqa: E402
from common.device import select_best_available_device  # noqa: E402

CONTEXT_LENGTH = 128       # how many characters the model can see at once
EMBEDDING_SIZE = 128
HEAD_COUNT = 4
BLOCK_COUNT = 4


class CausalSelfAttention(nn.Module):
    """Chapter 22's multi-head causal attention, batched.

    One linear layer computes Q, K, V for all heads at once (三 matrices
    stacked - purely an efficiency habit you will see in every real GPT);
    heads are separated by reshaping, attended independently, re-concatenated.
    """

    def __init__(self):
        super().__init__()
        self.query_key_value_projection = nn.Linear(EMBEDDING_SIZE, 3 * EMBEDDING_SIZE)
        self.output_projection = nn.Linear(EMBEDDING_SIZE, EMBEDDING_SIZE)

    def forward(self, token_features):
        batch_size, sequence_length, _ = token_features.shape
        head_size = EMBEDDING_SIZE // HEAD_COUNT

        queries, keys, values = self.query_key_value_projection(token_features).chunk(3, dim=-1)
        # (batch, seq, emb) -> (batch, heads, seq, head_size): each head gets
        # its own slice of the embedding to work in.
        def split_heads(tensor):
            return tensor.view(batch_size, sequence_length, HEAD_COUNT, head_size).transpose(1, 2)
        queries, keys, values = split_heads(queries), split_heads(keys), split_heads(values)

        # The fused kernel computes Chapter 22's exact formula (scores, scale,
        # causal mask, softmax, blend) - we verified that agreement there.
        attended = nn.functional.scaled_dot_product_attention(queries, keys, values, is_causal=True)

        attended = attended.transpose(1, 2).reshape(batch_size, sequence_length, EMBEDDING_SIZE)
        return self.output_projection(attended)


class TransformerBlock(nn.Module):
    """The block, exactly as Chapter 22 wrote it:
    communicate (attention), then compute (MLP), residuals around both."""

    def __init__(self):
        super().__init__()
        self.attention_norm = nn.LayerNorm(EMBEDDING_SIZE)
        self.attention = CausalSelfAttention()
        self.mlp_norm = nn.LayerNorm(EMBEDDING_SIZE)
        self.mlp = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE, 4 * EMBEDDING_SIZE),
            nn.GELU(),   # ReLU's smooth cousin - the transformer convention
            nn.Linear(4 * EMBEDDING_SIZE, EMBEDDING_SIZE),
        )

    def forward(self, token_features):
        token_features = token_features + self.attention(self.attention_norm(token_features))
        token_features = token_features + self.mlp(self.mlp_norm(token_features))
        return token_features


class MiniGPT(nn.Module):
    """Token embeddings + positional embeddings + blocks + next-token head."""

    def __init__(self, vocabulary_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocabulary_size, EMBEDDING_SIZE)
        self.position_embedding = nn.Embedding(CONTEXT_LENGTH, EMBEDDING_SIZE)
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(BLOCK_COUNT)])
        self.final_norm = nn.LayerNorm(EMBEDDING_SIZE)
        self.next_token_head = nn.Linear(EMBEDDING_SIZE, vocabulary_size)

    def forward(self, token_ids):
        sequence_length = token_ids.shape[1]
        positions = torch.arange(sequence_length, device=token_ids.device)
        # Attention is order-blind, so each token's vector carries both WHAT
        # it is (token embedding) and WHERE it sits (position embedding).
        token_features = self.token_embedding(token_ids) + self.position_embedding(positions)
        token_features = self.final_norm(self.blocks(token_features))
        return self.next_token_head(token_features)

    @torch.no_grad()
    def generate(self, token_ids, new_token_count, temperature=0.8):
        """Autoregressive sampling: predict, sample, append, repeat.

        Arguments:
            token_ids: (1, prompt_length) starting context.
            new_token_count: how many tokens to add.
            temperature: logit divisor - lower is safer, higher wilder.
        """
        for _ in range(new_token_count):
            visible_context = token_ids[:, -CONTEXT_LENGTH:]
            logits = self(visible_context)[:, -1]
            probabilities = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, 1)
            token_ids = torch.cat([token_ids, next_token], dim=1)
        return token_ids


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
    corpus = torch.tensor([character_to_index[c] for c in text], dtype=torch.long)

    # Held-out split (Chapter 12): the last 10% is never trained on.
    split_point = int(len(corpus) * 0.9)
    training_corpus, validation_corpus = corpus[:split_point], corpus[split_point:]

    model = MiniGPT(len(characters)).to(device)
    print(f"MiniGPT: {sum(p.numel() for p in model.parameters()):,} parameters, "
          f"{BLOCK_COUNT} blocks, {HEAD_COUNT} heads, context {CONTEXT_LENGTH}")

    def sample(prompt="ROMEO:", length=300):
        prompt_ids = torch.tensor([[character_to_index[c] for c in prompt]], device=device)
        generated = model.generate(prompt_ids, length)
        return "".join(characters[i] for i in generated[0].tolist())

    print("\nBefore training (an untrained GPT babbles uniformly):")
    print("-" * 60)
    print(sample(length=150))
    print("-" * 60)

    def get_batch(source, generator):
        starts = torch.randint(0, len(source) - CONTEXT_LENGTH - 1, (64,), generator=generator)
        inputs = torch.stack([source[s:s + CONTEXT_LENGTH] for s in starts])
        targets = torch.stack([source[s + 1:s + CONTEXT_LENGTH + 1] for s in starts])
        return inputs.to(device), targets.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    loss_function = nn.CrossEntropyLoss()
    batch_generator = torch.Generator().manual_seed(42)
    validation_generator = torch.Generator().manual_seed(999)

    print(f"\nTraining for {total_steps} steps...")
    print("   step    train loss   validation loss   seconds")
    training_start = time.perf_counter()
    for step_number in range(1, total_steps + 1):
        inputs, targets = get_batch(training_corpus, batch_generator)
        logits = model(inputs)
        loss = loss_function(logits.reshape(-1, len(characters)), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step_number in (1, 200, 1000, 2000, total_steps):
            model.eval()
            with torch.no_grad():
                validation_inputs, validation_targets = get_batch(validation_corpus, validation_generator)
                validation_loss = loss_function(
                    model(validation_inputs).reshape(-1, len(characters)),
                    validation_targets.reshape(-1),
                )
            model.train()
            elapsed = time.perf_counter() - training_start
            print(f"  {step_number:>5}   {loss.item():>10.4f}   {validation_loss.item():>15.4f}   {elapsed:>7.0f}")

    # Perplexity: e^loss - "the model is as uncertain as a fair choice among
    # this many characters". A friendlier number than raw loss.
    print(f"\nFinal validation perplexity: {math.exp(validation_loss.item()):.1f} "
          f"(untrained would be {len(characters)})")

    print("\nAfter training:")
    print("-" * 60)
    print(sample())
    print("-" * 60)


if __name__ == "__main__":
    main()
