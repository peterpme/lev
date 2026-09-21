"""Train the first Banking77-only Lev model."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import augment, build_banking77, materialize
from .model import DecisionModel, encode, load_tokenizer


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def scheduler_pct_start(total_updates: int) -> float:
    """Avoid OneCycleLR's zero-length warmup edge case in tiny learning runs."""
    return 0.3 if total_updates <= 10 else 0.1


def save_checkpoint(
    model: DecisionModel,
    tokenizer: object,
    output: Path,
    args: argparse.Namespace,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(output)
    torch.save(
        {
            "head": model.head.state_dict(),
            "base": args.base,
            "lora": args.lora,
            "args": vars(args),
        },
        output / "head.pt",
    )
    tokenizer.save_pretrained(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--n_per_source", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora", type=int, default=16)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("runs/banking77-smoke"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = choose_device(args.device)
    tokenizer = load_tokenizer(args.base)
    model = DecisionModel(args.base, device, lora_rank=args.lora)
    trainable = sum(parameter.numel() for parameter in model.trainable_parameters())
    print(f"device={device} trainable_params={trainable / 1e6:.2f}M", flush=True)

    requests = build_banking77(args.n_per_source, "train", args.seed)
    print(f"training_requests={len(requests)} source=banking77", flush=True)

    if args.dry_run:
        record = materialize(requests[0])
        encoded = encode(tokenizer, record)
        model.eval()
        probabilities = model.probabilities(encoded)[0]
        print(
            json.dumps(
                {
                    "sequence_tokens": len(encoded["ids"]),
                    "options": len(probabilities),
                    "probability_sum": float(probabilities.sum()),
                    "label": encoded["labels"][0],
                },
                indent=2,
            ),
            flush=True,
        )
        return

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(), lr=args.lr, weight_decay=0.01
    )
    updates_per_epoch = math.ceil(len(requests) / args.accum)
    total_updates = max(args.epochs * updates_per_epoch, 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_updates,
        pct_start=scheduler_pct_start(total_updates),
    )

    model.train()
    optimizer.zero_grad()
    started = time.time()
    update = 0
    seen = 0
    running_loss = 0.0
    for epoch in range(args.epochs):
        rng.shuffle(requests)
        for index, request in enumerate(requests):
            record = materialize(augment(request, rng))
            try:
                encoded = encode(tokenizer, record)
            except ValueError as error:
                print(f"skip: {error}", flush=True)
                continue

            logits = model(encoded)
            losses = [
                F.cross_entropy(
                    question_logits[None],
                    torch.tensor([question["label"]], device=model.device),
                )
                for question_logits, question in zip(logits, record["questions"])
            ]
            loss = sum(losses) / len(losses)
            (loss / args.accum).backward()
            running_loss += loss.item()
            seen += 1

            should_update = (index + 1) % args.accum == 0 or index + 1 == len(requests)
            if not should_update:
                continue
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            update += 1
            if device == "mps":
                torch.mps.empty_cache()
            if update % args.log_every == 0:
                elapsed = time.time() - started
                print(
                    f"epoch={epoch + 1}/{args.epochs} "
                    f"update={update}/{total_updates} "
                    f"loss={running_loss / seen:.4f} "
                    f"seconds_per_record={elapsed / seen:.2f}",
                    flush=True,
                )

    save_checkpoint(model, tokenizer, args.out, args)
    print(f"saved={args.out}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
