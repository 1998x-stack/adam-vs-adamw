"""Command-line interface for all four experiment tracks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from .core import METHODS, choose_device, configure_logging, write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adam+L2 versus AdamW reproducible experiments")
    p.add_argument("--experiment", choices=["quadratic", "moons", "cifar10", "wikitext2"], required=True)
    p.add_argument("--methods", default="adam,adam_l2,adamw")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--lr", type=float, default=None, help="Default depends on experiment")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--steps", type=int, default=None, help="Optimizer steps (non-CIFAR)")
    p.add_argument("--epochs", type=int, default=20, help="CIFAR epochs")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--device", default="auto")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--limit-train", type=int, default=0, help="For quick CIFAR experimentation")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--max-eval-batches", type=int, default=16)
    p.add_argument("--sweep", action="store_true", help="Validation-only LR x decay sweep (moons)")
    p.add_argument("--sweep-lrs", default="0.001,0.003,0.01")
    p.add_argument("--sweep-decays", default="0,0.0001,0.001,0.01,0.1")
    args = p.parse_args(argv)
    args.methods = [m.strip() for m in args.methods.split(",")]
    if not args.methods or any(m not in METHODS for m in args.methods):
        p.error(f"--methods must be comma-separated values from {METHODS}")
    args.seeds = [int(s.strip()) for s in args.seeds.split(",")]
    if not args.seeds or args.eval_every <= 0 or (args.steps is not None and args.steps <= 0):
        p.error("seeds must be nonempty, and step intervals must be positive")
    if args.weight_decay < 0 or args.epochs <= 0:
        p.error("weight decay must be nonnegative, epochs positive")
    if args.sweep and args.experiment != "moons":
        p.error("--sweep is currently implemented for the moons experiment only")
    return args


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    from .experiments import run_cifar, run_moons, run_quadratic, run_wikitext

    name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out_dir) / args.experiment / name
    configure_logging(out)
    write_json(out / "config.json", vars(args))
    logger.info("experiment={} methods={} seeds={}", args.experiment, args.methods, args.seeds)
    device = choose_device(args.device)

    if args.experiment == "quadratic":
        run_quadratic(args, out, device)
    elif args.experiment == "moons":
        run_moons(args, out, device)
    elif args.experiment == "cifar10":
        run_cifar(args, out, device)
    else:
        run_wikitext(args, out, device)
    logger.success("Completed: {}", out.resolve())
    return out


if __name__ == "__main__":
    main()
