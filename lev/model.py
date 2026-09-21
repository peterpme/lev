"""Kev-like Qwen backbone, block-causal mask, and pointer readout."""

from __future__ import annotations

import math
import re
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

SPECIAL = [
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|fim_suffix|>",
]
MAX_STATE = 384
MAX_BRANCH = 1024
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def load_tokenizer(name: str) -> Any:
    return AutoTokenizer.from_pretrained(name)


def user_tokens(tokenizer: Any, text: str) -> list[int]:
    """Prevent user text from creating Lev's control tokens."""
    safe = _SPECIAL_RE.sub(r"<¦\1¦>", text)
    return tokenizer(safe, add_special_tokens=False).input_ids


def encode(
    tokenizer: Any,
    record: dict[str, Any],
    max_state: int = MAX_STATE,
    max_branch: int = MAX_BRANCH,
) -> dict[str, Any]:
    """Pack shared state followed by isolated question branches."""
    state = [tokenizer.convert_tokens_to_ids(SPECIAL[0])]
    state += user_tokens(tokenizer, record["state"])[: max_state - 1]
    ids = list(state)
    segments = [0] * len(state)
    positions = list(range(len(state)))
    question_id, option_id, close_id, decide_id = (
        tokenizer.convert_tokens_to_ids(token) for token in SPECIAL[1:]
    )
    decide_indices: list[int] = []
    option_indices: list[list[int]] = []

    for segment, question in enumerate(record["questions"], start=1):
        branch = [question_id] + user_tokens(tokenizer, question["instr"])
        option_ends: list[int] = []
        for option in question["options"]:
            branch += [option_id] + user_tokens(tokenizer, option) + [close_id]
            option_ends.append(len(branch) - 1)
        branch.append(decide_id)
        if len(branch) > max_branch - len(state):
            raise ValueError(f"branch too long: {len(branch)}")

        base = len(ids)
        ids += branch
        segments += [segment] * len(branch)
        positions += list(range(len(state), len(state) + len(branch)))
        decide_indices.append(base + len(branch) - 1)
        option_indices.append([base + index for index in option_ends])

    return {
        "ids": ids,
        "seg": segments,
        "pos": positions,
        "decide_idx": decide_indices,
        "opt_idx": option_indices,
        "labels": [question["label"] for question in record["questions"]],
    }


def branch_mask(
    segments: list[int],
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Allow causal attention to shared state and the current question only."""
    segment_tensor = torch.tensor(segments, device=device)
    length = len(segments)
    causal = torch.tril(torch.ones(length, length, dtype=torch.bool, device=device))
    same_branch = (
        segment_tensor[None, :] == segment_tensor[:, None]
    ) | (segment_tensor[None, :] == 0)
    allowed = causal & same_branch
    return torch.zeros(length, length, dtype=dtype, device=device).masked_fill(
        ~allowed, torch.finfo(dtype).min
    )[None, None]


class PointerHead(nn.Module):
    def __init__(self, hidden_size: int, pointer_size: int = 256):
        super().__init__()
        self.query = nn.Linear(hidden_size, pointer_size)
        self.key = nn.Linear(hidden_size, pointer_size)
        self.scale = 1 / math.sqrt(pointer_size)

    def forward(
        self, decide_hidden: torch.Tensor, option_hidden: torch.Tensor
    ) -> torch.Tensor:
        return (self.key(option_hidden) @ self.query(decide_hidden)) * self.scale


class DecisionModel(nn.Module):
    def __init__(
        self,
        name: str,
        device: torch.device | str,
        lora_rank: int | None = 16,
    ):
        super().__init__()
        causal_lm = AutoModelForCausalLM.from_pretrained(
            name,
            dtype=torch.float32,
            attn_implementation="eager",
        )
        self.backbone = causal_lm.model
        del causal_lm

        if lora_rank:
            from peft import LoraConfig, get_peft_model

            config = LoraConfig(
                r=lora_rank,
                lora_alpha=2 * lora_rank,
                lora_dropout=0.05,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )
            self.backbone = get_peft_model(self.backbone, config)

        self.head = PointerHead(self.backbone.config.hidden_size)
        self.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def hidden(self, encoded: dict[str, Any]) -> torch.Tensor:
        input_ids = torch.tensor([encoded["ids"]], device=self.device)
        position_ids = torch.tensor([encoded["pos"]], device=self.device)
        attention_mask = branch_mask(encoded["seg"], self.device)
        return self.backbone(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[0]

    def forward(self, encoded: dict[str, Any]) -> list[torch.Tensor]:
        hidden = self.hidden(encoded)
        return [
            self.head(
                hidden[decide_index],
                hidden[torch.tensor(option_indices, device=self.device)],
            )
            for decide_index, option_indices in zip(
                encoded["decide_idx"], encoded["opt_idx"]
            )
        ]

    @torch.no_grad()
    def probabilities(self, encoded: dict[str, Any]) -> list[torch.Tensor]:
        return [F.softmax(logits, dim=-1).cpu() for logits in self(encoded)]

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]
