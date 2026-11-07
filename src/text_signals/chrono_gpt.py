"""ChronoGPT-Instruct, a chronologically consistent LLM, for look-ahead-free scoring (§6.7).

He, Lv, Manela and Wu (2025), "Chronologically Consistent Large Language Models",
arXiv:2502.21206. One model per vintage τ (31 December of 1999, 2000, ..., 2024), each trained
only on text dated before τ. A statement released in year Y is read by the vintage of Y − 1, so
no statement is read by a model that has seen what followed it.

The model code below is the forward pass of ``ChronoGPT_instruct.py`` from the model repositories
(``manelalab/chrono-gpt-instruct-v1-<τ>``, MIT licence), copied so that no downloaded code is
executed; only the inference path is kept. Weights are loaded with ``weights_only=True``.

As an instruction follower the model cannot do this task (protocol §6.7), so it is used as a
language model: a sentence is followed by "As a result, short-term interest rates are likely
to" and its score is log P(" rise") − log P(" fall") for the next token, minus the vintage's mean
on fixed content-free anchor sentences. A statement's score is the mean over its sentences.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE, NUM_LAYERS, NUM_HEADS, MODEL_DIM = 50304, 52, 12, 1536
CONTEXT = 1792
EOT = 50256


def _norm(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.size(-1),))


class _CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.type_as(x))


class _Rotary(nn.Module):
    cos: torch.Tensor
    sin: torch.Tensor

    def __init__(self, dim: int, max_seq_len: int = CONTEXT) -> None:
        super().__init__()
        freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        freq = torch.cat([freq, freq.new_zeros(dim // 4)])
        theta = torch.einsum("i,j -> ij", torch.arange(max_seq_len, dtype=torch.float32), freq)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cos = self.cos[None, : x.size(-3), None, :]
        sin = self.sin[None, : x.size(-3), None, :]
        x1, x2 = x.float().chunk(2, dim=-1)
        return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), 3).type_as(x)


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.c_q, self.c_k, self.c_v = (_CastedLinear(dim, dim) for _ in range(3))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = _Rotary(self.head_dim)
        self.c_proj = _CastedLinear(dim, dim)

    def forward(self, x: torch.Tensor, ve: torch.Tensor | None) -> torch.Tensor:
        b, t = x.size(0), x.size(1)
        q = self.c_q(x).view(b, t, self.num_heads, self.head_dim)
        k = self.c_k(x).view(b, t, self.num_heads, self.head_dim)
        v = self.c_v(x).view(b, t, self.num_heads, self.head_dim)
        v = self.lambdas[0] * v + (self.lambdas[1] * ve.view_as(v) if ve is not None else 0)
        q, k = self.rotary(_norm(q)), self.rotary(_norm(k))
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        )
        out: torch.Tensor = self.c_proj(y.transpose(1, 2).contiguous().view(b, t, -1))
        return out


class _MLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.c_fc, self.c_proj = _CastedLinear(dim, 4 * dim), _CastedLinear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.c_proj(F.relu(self.c_fc(x)).square())
        return out


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.attn, self.mlp = _Attention(dim, num_heads), _MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(self, x: torch.Tensor, ve: torch.Tensor | None, x0: torch.Tensor) -> torch.Tensor:
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = x + self.attn(_norm(x), ve)
        out: torch.Tensor = x + self.mlp(_norm(x))
        return out


class _ValueEmbedding(nn.Module):
    def __init__(self, vocab: int, dim: int, num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.embed = nn.ModuleList([nn.Embedding(vocab, dim) for _ in range(3)])

    def forward(self, ids: torch.Tensor, dtype: torch.dtype) -> list[torch.Tensor | None]:
        base = [emb(ids).to(dtype) for emb in self.embed]
        half = self.num_layers // 2
        enc: list[torch.Tensor | None] = [base[i] if i < 3 else None for i in range(half)]
        dec: list[torch.Tensor | None] = [
            base[i - (half - 3)] if i >= half - 3 else None for i in range(half)
        ]
        return enc + dec


class ChronoGPT(nn.Module):
    """The inference path of ChronoGPT-Instruct (batch of one sequence)."""

    def __init__(self, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.dtype = dtype
        self.embed = nn.Embedding(VOCAB_SIZE, MODEL_DIM)
        self.blocks = nn.ModuleList([_Block(MODEL_DIM, NUM_HEADS) for _ in range(NUM_LAYERS)])
        self.value_embeds = _ValueEmbedding(VOCAB_SIZE, MODEL_DIM, NUM_LAYERS)
        self.lm_head = _CastedLinear(MODEL_DIM, VOCAB_SIZE)
        self.num_encoder_layers = NUM_LAYERS // 2
        self.skip_weights = nn.Parameter(torch.ones(NUM_LAYERS - self.num_encoder_layers))

    @torch.inference_mode()
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Logits of the last position for a 1-D tensor of token ids."""
        ids = ids[-CONTEXT:]
        x0 = _norm(self.embed(ids[None]).to(self.dtype))
        x = x0
        ve = [None if v is None else v[None] for v in self.value_embeds(ids, self.dtype)]
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, ve[i], x0)
            skips.append(x)
        for i in range(NUM_LAYERS - self.num_encoder_layers):
            x = x + self.skip_weights[i] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, ve[self.num_encoder_layers + i], x0)
        logits = self.lm_head(_norm(x)[:, -1])
        return (15 * torch.tanh(logits / 15)).float()[0]


def load(weights: str, dtype: torch.dtype = torch.bfloat16) -> ChronoGPT:
    model = ChronoGPT(dtype)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if [m for m in missing if not m.endswith(("cos", "sin"))] or unexpected:
        raise ValueError(f"weights do not match the model: {missing} {unexpected}")
    return model.eval()


PROMPT_VERSION = "chrono-rise-fall-v1"
CONTINUATION = " As a result, short-term interest rates are likely to"
RISE, FALL = 4485, 2121  # GPT-2 tokens " rise" and " fall"
# Content-free sentences; their mean log-odds is each vintage's offset (protocol §6.7).
ANCHORS = (
    "The Committee met today.",
    "Voting for the action were members of the Committee.",
    "The statement was released at the end of the meeting.",
    "The Committee will meet again as scheduled.",
    "Members of the Committee discussed the matter.",
)


def log_odds(model: ChronoGPT, encode: Callable[[str], list[int]], sentence: str) -> float:
    """log P(" rise") − log P(" fall") for the next token after the sentence and the cue."""
    ids = torch.tensor(encode(sentence + CONTINUATION), dtype=torch.long)
    logp = torch.log_softmax(model(ids), dim=0)
    return float(logp[RISE] - logp[FALL])


def anchor_offset(model: ChronoGPT, encode: Callable[[str], list[int]]) -> float:
    return sum(log_odds(model, encode, a) for a in ANCHORS) / len(ANCHORS)
