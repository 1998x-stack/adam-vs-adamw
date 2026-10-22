"""Experimental tracks with paired initialization and matching data order.

Benchmark rule: raw task loss is always used for evaluation. L2 is implemented
by torch.optim.Adam(weight_decay=λ), which adds λθ to the gradient *before*
the moment updates; AdamW performs decoupled weight decay.
"""
from __future__ import annotations

import copy
import math
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from torch import nn

from .core import (
    make_optimizer, param_norm, seed_everything, step_with_metrics, write_csv, write_json
)
from .plots import plot_boundary, plot_curves, plot_quadratic, plot_sweep


def _summary(method: str, seed: int, **metrics: float) -> dict:
    return {"method": method, "seed": seed, **{k: float(v) for k, v in metrics.items()}}


def run_quadratic(args: Namespace, out: Path, device: torch.device) -> None:
    """Different coordinate curvature exposes the optimizer preconditioner."""
    lr = args.lr if args.lr is not None else 0.05
    steps = args.steps or 100
    rows, summary, traces = [], [], {}
    start = (-2.0, 2.5)
    target = torch.tensor([2.0, 1.0], device=device)
    curvature = torch.tensor([1.0, 20.0], device=device)
    for seed in args.seeds:
        for method in args.methods:
            seed_everything(seed)
            model = nn.Module()
            model.register_parameter(
                "theta", nn.Parameter(torch.tensor(start, device=device))
            )
            opt = make_optimizer(model, method, lr, args.weight_decay, exclude_norm_bias=False)
            trace = [model.theta.detach().cpu().numpy().copy().tolist()]
            for step in range(steps + 1):
                loss = 0.5 * ((model.theta - target).square() * curvature).sum()
                if step == 0 or step % args.eval_every == 0 or step == steps:
                    rows.append({
                        "experiment": "quadratic", "method": method, "seed": seed,
                        "step": step, "train_loss": float(loss.detach()),
                        "weight_norm": param_norm(model.parameters()),
                        "theta_0": float(model.theta[0]), "theta_1": float(model.theta[1]),
                    })
                if step == steps:
                    break
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                trace.append(model.theta.detach().cpu().numpy().copy().tolist())
            if seed == args.seeds[0]:
                traces[method] = np.asarray(trace)
            summary.append(_summary(method, seed, final_task_loss=float(loss.detach()),
                                    final_weight_norm=param_norm(model.parameters())))
            logger.info("quadratic {} seed={} loss={:.5f}", method, seed, loss.item())
    write_csv(out / "metrics.csv", rows)
    write_csv(out / "summary.csv", summary)
    write_json(out / "summary.json", summary)
    plot_quadratic(out, traces, start=start, decay=args.weight_decay)
    plot_curves(out, rows, ["train_loss", "weight_norm"])


class MoonMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _moon_data(seed: int, device: torch.device) -> tuple:
    features, labels = make_moons(n_samples=2360, noise=0.30, random_state=seed)
    perm = np.random.default_rng(seed + 777).permutation(len(features))
    a, b, c = perm[:160], perm[160:360], perm[360:]
    scaler = StandardScaler().fit(features[a])
    x = torch.from_numpy(scaler.transform(features).astype("float32")).to(device)
    y = torch.tensor(labels, dtype=torch.float32, device=device)
    return (x[a], y[a]), (x[b], y[b]), (x[c], y[c])


@torch.no_grad()
def _binary_metrics(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    model.eval()
    logits = model(x)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, y)
    accuracy = ((logits >= 0).float() == y).float().mean()
    return float(loss), float(accuracy)


def _train_moon(
    seed: int, method: str, lr: float, decay: float, steps: int, eval_every: int,
    batch_size: int, device: torch.device, data: tuple, *, test_at_end: bool
) -> tuple[list[dict], dict, MoonMLP]:
    seed_everything(seed)
    model = MoonMLP().to(device)
    opt = make_optimizer(model, method, lr, decay)
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = data
    batch_rng = torch.Generator().manual_seed(seed + 900)
    params = list(model.parameters())
    rows = []
    update_norm = update_ratio = 0.0
    for step in range(steps + 1):
        if step == 0 or step % eval_every == 0 or step == steps:
            train_loss, train_acc = _binary_metrics(model, train_x, train_y)
            val_loss, val_acc = _binary_metrics(model, val_x, val_y)
            rows.append({
                "experiment": "moons", "method": method, "seed": seed, "step": step,
                "learning_rate": lr, "decay": decay,
                "train_loss": train_loss, "val_loss": val_loss,
                "train_accuracy": train_acc, "val_accuracy": val_acc,
                "weight_norm": param_norm(params),
                "update_norm": update_norm, "update_ratio": update_ratio,
            })
        if step == steps:
            break
        model.train()
        selected = torch.randint(0, len(train_x), (batch_size,), generator=batch_rng).to(device)
        loss = nn.functional.binary_cross_entropy_with_logits(
            model(train_x[selected]), train_y[selected]
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        update_norm, update_ratio = step_with_metrics(opt, params)
    final = rows[-1]
    scores = dict(val_loss=final["val_loss"], val_accuracy=final["val_accuracy"],
                  train_loss=final["train_loss"], weight_norm=final["weight_norm"])
    if test_at_end:
        test_loss, test_acc = _binary_metrics(model, test_x, test_y)
        scores.update(test_loss=test_loss, test_accuracy=test_acc)
    return rows, _summary(method, seed, **scores), model


def run_moons(args: Namespace, out: Path, device: torch.device) -> None:
    if args.sweep:
        run_moon_sweep(args, out, device)
        return
    lr = args.lr if args.lr is not None else 0.003
    steps = args.steps or 1200
    batch_size = args.batch_size or 32
    rows, summary = [], []
    for seed in args.seeds:
        data = _moon_data(seed, device)
        for method in args.methods:
            history, record, model = _train_moon(
                seed, method, lr, args.weight_decay, steps, args.eval_every,
                batch_size, device, data, test_at_end=True
            )
            rows.extend(history)
            summary.append(record)
            logger.info("moons {} seed={} val_loss={:.4f} test_acc={:.3f}",
                        method, seed, record["val_loss"], record["test_accuracy"])
            if seed == args.seeds[0]:
                plot_boundary(out, data, model, method)
    write_csv(out / "metrics.csv", rows)
    write_csv(out / "summary.csv", summary)
    write_json(out / "summary.json", summary)
    plot_curves(out, rows, [
        "train_loss", "val_loss", "val_accuracy", "weight_norm", "update_ratio",
    ])


def run_moon_sweep(args: Namespace, out: Path, device: torch.device) -> None:
    """Only validation metrics are accessed; test split is never touched."""
    lrs = [float(v) for v in args.sweep_lrs.split(",")]
    decays = [float(v) for v in args.sweep_decays.split(",")]
    steps = args.steps or 600
    records = []
    for seed in args.seeds:
        data = _moon_data(seed, device)
        for method in args.methods:
            for lr in lrs:
                for decay in decays:
                    _, record, _ = _train_moon(
                        seed, method, lr, decay, steps, max(steps, 1),
                        args.batch_size or 32, device, data, test_at_end=False
                    )
                    records.append({"method": method, "seed": seed, "lr": lr,
                                    "decay": decay, "val_loss": record["val_loss"],
                                    "val_accuracy": record["val_accuracy"]})
                    logger.info("sweep {} s={} lr={} wd={} val={:.4f}",
                                method, seed, lr, decay, record["val_loss"])
    write_csv(out / "sweep.csv", records)
    plot_sweep(out, records)


def _classify_eval(model: nn.Module, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    loss_total = correct = count = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss_total += float(nn.functional.cross_entropy(logits, labels, reduction="sum"))
            correct += int((logits.argmax(-1) == labels).sum())
            count += int(labels.numel())
    return loss_total / count, correct / count


def run_cifar(args: Namespace, out: Path, device: torch.device) -> None:
    """CIFAR-10: official 50k train -> 40k train/10k validation; 10k official test."""
    try:
        from torchvision import datasets, models, transforms
        from torch.utils.data import DataLoader, Subset
    except ImportError as exc:
        raise RuntimeError("Install optional vision deps: pip install -e '.[vision]'") from exc

    lr = args.lr if args.lr is not None else 0.001
    batch_size = args.batch_size or 128
    base = Path(args.data_dir)
    normalize = transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)
    )
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize,
    ])
    eval_tf = transforms.Compose([transforms.ToTensor(), normalize])
    train_base = datasets.CIFAR10(base, train=True, transform=train_tf, download=True)
    val_base = datasets.CIFAR10(base, train=True, transform=eval_tf, download=False)
    test_ds = datasets.CIFAR10(base, train=False, transform=eval_tf, download=True)
    train_idx, val_idx = torch.randperm(50000, generator=torch.Generator().manual_seed(2026)).split(
        [40000, 10000]
    )
    # Identical official split across all optimizer methods and random seeds.
    if args.limit_train:
        train_idx = train_idx[:args.limit_train]
    train_set = Subset(train_base, train_idx.tolist())
    val_set = Subset(val_base, val_idx.tolist())
    loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)
    histories, summary = [], []

    def model_factory():
        net = models.resnet18(weights=None, num_classes=10)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        net.maxpool = nn.Identity()
        return net.to(device)

    for seed in args.seeds:
        for method in args.methods:
            seed_everything(seed)
            model = model_factory()
            opt = make_optimizer(model, method, lr, args.weight_decay)
            train_loader = DataLoader(
                train_set, shuffle=True,
                generator=torch.Generator().manual_seed(seed + 2000), **loader_args
            )
            for epoch in range(1, args.epochs + 1):
                model.train()
                total_loss = total_count = correct = 0
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    opt.zero_grad(set_to_none=True)
                    logits = model(images)
                    loss = nn.functional.cross_entropy(logits, labels)
                    loss.backward()
                    opt.step()
                    n = labels.numel()
                    total_loss += float(loss.detach()) * n
                    total_count += n
                    correct += int((logits.argmax(-1) == labels).sum())
                val_loss, val_accuracy = _classify_eval(model, val_loader, device)
                row = {
                    "experiment": "cifar10", "method": method, "seed": seed,
                    "step": epoch, "train_loss": total_loss / total_count,
                    "train_accuracy": correct / total_count, "val_loss": val_loss,
                    "val_accuracy": val_accuracy, "weight_norm": param_norm(model.parameters()),
                }
                histories.append(row)
                logger.info("cifar {} seed={} epoch={}/{} val_acc={:.4f}",
                            method, seed, epoch, args.epochs, val_accuracy)
            test_loss, test_accuracy = _classify_eval(model, test_loader, device)
            summary.append(_summary(
                method, seed, val_loss=val_loss, val_accuracy=val_accuracy,
                test_loss=test_loss, test_accuracy=test_accuracy,
                weight_norm=param_norm(model.parameters())
            ))
    write_csv(out / "metrics.csv", histories)
    write_csv(out / "summary.csv", summary)
    write_json(out / "summary.json", summary)
    plot_curves(out, histories, ["train_loss", "val_loss", "val_accuracy", "weight_norm"])


class TinyCausalTransformer(nn.Module):
    """A small causal Transformer trained from scratch; no pretrained model weights."""
    def __init__(self, vocab_size: int, seq_len: int, d_model: int = 128, layers: int = 2):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        block = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, dim_feedforward=4 * d_model,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        _, length = ids.shape
        if length > self.pos.num_embeddings:
            raise ValueError("input exceeds configured sequence length")
        h = self.token(ids) + self.pos(torch.arange(length, device=ids.device))[None]
        causal_mask = torch.triu(
            torch.ones(length, length, device=ids.device, dtype=torch.bool), diagonal=1
        )
        return self.head(self.norm(self.transformer(h, mask=causal_mask)))


def _language_chunks(raw: list[str], tokenizer, seq_len: int) -> torch.Tensor:
    text = "\n".join(line for line in raw if line.strip())
    ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False), dtype=torch.long)
    chunk = seq_len + 1
    available = len(ids) // chunk
    if available == 0:
        raise ValueError("Not enough data for one training sequence")
    return ids[:available * chunk].reshape(available, chunk)


@torch.no_grad()
def _language_eval(
    model: TinyCausalTransformer, chunks: torch.Tensor, device: torch.device,
    max_batches: int, batch_size: int,
) -> float:
    model.eval()
    loss_sum = count = 0
    for part in chunks[:max_batches * batch_size].split(batch_size):
        part = part.to(device)
        logits = model(part[:, :-1])
        target = part[:, 1:]
        loss_sum += float(nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="sum"
        ))
        count += target.numel()
    return loss_sum / count


def run_wikitext(args: Namespace, out: Path, device: torch.device) -> None:
    """WikiText-2 raw, GPT-2 *tokenizer only*, small random-initialized causal model."""
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install NLP dependencies: pip install -e '.[nlp]'") from exc

    lr = args.lr if args.lr is not None else 0.0003
    steps = args.steps or 500
    batch_size = args.batch_size or 16
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", cache_dir=args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=args.data_dir)
    train = _language_chunks(data["train"]["text"], tokenizer, args.seq_len)
    valid = _language_chunks(data["validation"]["text"], tokenizer, args.seq_len)
    test = _language_chunks(data["test"]["text"], tokenizer, args.seq_len)
    histories, summary = [], []
    for seed in args.seeds:
        for method in args.methods:
            seed_everything(seed)
            model = TinyCausalTransformer(tokenizer.vocab_size, args.seq_len).to(device)
            opt = make_optimizer(model, method, lr, args.weight_decay, betas=(0.9, 0.95))
            sampler = torch.Generator().manual_seed(seed + 3000)
            params = list(model.parameters())
            for step in range(1, steps + 1):
                model.train()
                selected = torch.randint(0, len(train), (batch_size,), generator=sampler)
                part = train[selected].to(device)
                logits = model(part[:, :-1])
                targets = part[:, 1:]
                train_loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
                )
                opt.zero_grad(set_to_none=True)
                train_loss.backward()
                nn.utils.clip_grad_norm_(params, max_norm=1.0)
                opt.step()
                if step == 1 or step % args.eval_every == 0 or step == steps:
                    val_loss = _language_eval(
                        model, valid, device, args.max_eval_batches, batch_size
                    )
                    histories.append({
                        "experiment": "wikitext2", "method": method, "seed": seed,
                        "step": step, "train_batch_loss": float(train_loss.detach()),
                        "val_loss": val_loss, "val_perplexity": math.exp(min(40, val_loss)),
                        "weight_norm": param_norm(params),
                    })
                    logger.info("wikitext2 {} seed={} step={}/{} val_ppl={:.2f}",
                                method, seed, step, steps, math.exp(min(40, val_loss)))
            test_loss = _language_eval(model, test, device, args.max_eval_batches, batch_size)
            summary.append(_summary(
                method, seed, val_loss=val_loss, val_perplexity=math.exp(min(40, val_loss)),
                test_loss=test_loss, test_perplexity=math.exp(min(40, test_loss)),
                weight_norm=param_norm(params),
            ))
    write_csv(out / "metrics.csv", histories)
    write_csv(out / "summary.csv", summary)
    write_json(out / "summary.json", summary)
    plot_curves(out, histories, [
        "train_batch_loss", "val_loss", "val_perplexity", "weight_norm",
    ])
