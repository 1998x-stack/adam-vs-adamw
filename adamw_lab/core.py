"""Shared, auditable training primitives and experiment I/O."""
from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from loguru import logger

METHODS = ("adam", "adam_l2", "adamw")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_logging(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(lambda message: print(message, end=""), level="INFO", colorize=False)
    logger.add(output / "run.log", rotation="10 MB", retention=3, level="DEBUG")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_dataclass(payload) and not isinstance(payload, type):
        payload = asdict(payload)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def param_norm(params: Iterable[torch.nn.Parameter]) -> float:
    with torch.no_grad():
        return math.sqrt(sum(float(p.detach().float().square().sum()) for p in params))


def step_with_metrics(
    optimizer: torch.optim.Optimizer,
    params: list[torch.nn.Parameter],
) -> tuple[float, float]:
    """Return (absolute update L2 norm, update/previous-weight norm)."""
    with torch.no_grad():
        previous = [p.detach().clone() for p in params]
    optimizer.step()
    with torch.no_grad():
        update = math.sqrt(sum(
            float((p.detach().float() - old.float()).square().sum())
            for p, old in zip(params, previous)
        ))
        before = math.sqrt(sum(float(old.float().square().sum()) for old in previous))
    return update, update / max(before, 1e-12)


def parameter_groups(
    model: torch.nn.Module, decay: float, *, exclude_norm_bias: bool = True
) -> list[dict[str, Any]]:
    """Exactly the same decay mask for coupled Adam L2 and AdamW.

    Default policy: exclude vectors/scalars (biases and norm affine parameters).
    To decay a one-dimensional toy vector, pass exclude_norm_bias=False.
    """
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = exclude_norm_bias and (param.ndim < 2 or name.endswith(".bias"))
        (no_decay_params if no_decay else decay_params).append(param)
    assert len(decay_params) + len(no_decay_params) > 0
    return [
        {"params": decay_params, "weight_decay": float(decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def make_optimizer(
    model: torch.nn.Module,
    method: str,
    lr: float,
    decay: float,
    betas: tuple[float, float] = (0.9, 0.999),
    *,
    exclude_norm_bias: bool = True,
) -> torch.optim.Optimizer:
    if method not in METHODS:
        raise ValueError(f"Unknown optimizer: {method}")
    groups = parameter_groups(
        model, 0.0 if method == "adam" else decay, exclude_norm_bias=exclude_norm_bias
    )
    options = dict(lr=lr, betas=betas, eps=1e-8, foreach=False)
    # Adam(weight_decay=λ) applies COUPLED L2, adding λθ to the gradient
    # *before* updating moments; AdamW performs separate decoupled shrinkage.
    if method == "adamw":
        return torch.optim.AdamW(groups, **options)
    return torch.optim.Adam(groups, **options)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    logger.info("device={}", device)
    return device
