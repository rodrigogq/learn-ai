"""Chapter 24 - the mini-LLM model definition, shared by training, sampling,
and the Chapter 25 export.

This is Chapter 23's MiniGPT with its dimensions made configurable, so one
class covers a smoke-test toy and a multi-day model. Keeping the definition in
one importable file means the training script, the sampler, and the weight
exporter can never disagree about the architecture.

The MODEL_SIZES table is the chapter's menu of runs. VRAM/time estimates are
rough and assume float32; mixed precision (see the chapter) roughly halves the
memory.
"""

import math

import torch
from torch import nn

# name -> architecture and training defaults.
MODEL_SIZES = {
    # ~1 M params: seconds per hundred steps, for smoke tests.
    "tiny":   {"block_count": 4,  "embedding_size": 128, "head_count": 4,  "context_length": 128,
               "batch_size": 32, "learning_rate": 3e-4, "default_steps": 2000},
    # ~11 M params: a real small LM, ~1 hour on a laptop GPU.
    "small":  {"block_count": 6,  "embedding_size": 384, "head_count": 6,  "context_length": 256,
               "batch_size": 32, "learning_rate": 3e-4, "default_steps": 20000},
    # ~45 M params: comfortable on a 16 GB GPU / 64 GB Mac, several hours.
    "medium": {"block_count": 8,  "embedding_size": 640, "head_count": 10, "context_length": 512,
               "batch_size": 24, "learning_rate": 2.5e-4, "default_steps": 60000},
    # ~110 M params (GPT-2 scale): the multi-day run the chapter is named for.
    "large":  {"block_count": 12, "embedding_size": 768, "head_count": 12, "context_length": 512,
               "batch_size": 16, "learning_rate": 2e-4, "default_steps": 200000},
}


class CausalSelfAttention(nn.Module):
    """Chapter 22/23's multi-head causal attention, dimensions from the config."""

    def __init__(self, config):
        super().__init__()
        self.embedding_size = config["embedding_size"]
        self.head_count = config["head_count"]
        self.query_key_value_projection = nn.Linear(self.embedding_size, 3 * self.embedding_size)
        self.output_projection = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, token_features):
        batch_size, sequence_length, _ = token_features.shape
        head_size = self.embedding_size // self.head_count
        queries, keys, values = self.query_key_value_projection(token_features).chunk(3, dim=-1)

        def split_heads(tensor):
            return tensor.view(batch_size, sequence_length, self.head_count, head_size).transpose(1, 2)
        queries, keys, values = split_heads(queries), split_heads(keys), split_heads(values)

        attended = nn.functional.scaled_dot_product_attention(queries, keys, values, is_causal=True)
        attended = attended.transpose(1, 2).reshape(batch_size, sequence_length, self.embedding_size)
        return self.output_projection(attended)


class TransformerBlock(nn.Module):
    """Communicate (attention) then compute (MLP), residuals around each."""

    def __init__(self, config):
        super().__init__()
        embedding_size = config["embedding_size"]
        self.attention_norm = nn.LayerNorm(embedding_size)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.GELU(),
            nn.Linear(4 * embedding_size, embedding_size),
        )

    def forward(self, token_features):
        token_features = token_features + self.attention(self.attention_norm(token_features))
        token_features = token_features + self.mlp(self.mlp_norm(token_features))
        return token_features


class MiniLanguageModel(nn.Module):
    """The full decoder-only transformer language model."""

    def __init__(self, vocabulary_size, config):
        super().__init__()
        self.config = config
        embedding_size = config["embedding_size"]
        self.token_embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.position_embedding = nn.Embedding(config["context_length"], embedding_size)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config["block_count"])])
        self.final_norm = nn.LayerNorm(embedding_size)
        self.next_token_head = nn.Linear(embedding_size, vocabulary_size, bias=False)
        # Weight tying: the input embedding and output projection share one
        # matrix. Standard in GPT-2 - it saves parameters and helps quality,
        # since "which token" is the same question at both ends.
        self.next_token_head.weight = self.token_embedding.weight
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids):
        sequence_length = token_ids.shape[1]
        positions = torch.arange(sequence_length, device=token_ids.device)
        token_features = self.token_embedding(token_ids) + self.position_embedding(positions)
        return self.next_token_head(self.final_norm(self.blocks(token_features)))

    @torch.no_grad()
    def generate(self, token_ids, new_token_count, temperature=0.8, top_k=40):
        """Autoregressive sampling with temperature and top-k (Chapter 23)."""
        context_length = self.config["context_length"]
        for _ in range(new_token_count):
            logits = self(token_ids[:, -context_length:])[:, -1] / temperature
            if top_k is not None:
                kth_value = torch.topk(logits, top_k, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth_value, float("-inf"))
            probabilities = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, 1)
            token_ids = torch.cat([token_ids, next_token], dim=1)
        return token_ids
