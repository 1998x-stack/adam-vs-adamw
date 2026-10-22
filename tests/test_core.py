"""Tests for optimizer semantics and reproducibility."""
import copy

import pytest
import torch
from torch import nn

from adamw_lab.core import make_optimizer, parameter_groups, seed_everything
from adamw_lab.experiments import _moon_data, MoonMLP


def test_weight_decay_first_step_is_decoupled():
    def start(method):
        model = nn.Module()
        model.register_parameter("theta", nn.Parameter(torch.tensor([1.0, 10.0])))
        optim = make_optimizer(model, method, lr=0.1, decay=0.1, exclude_norm_bias=False)
        optim.zero_grad(set_to_none=True)
        model.theta.grad = torch.zeros_like(model.theta)
        optim.step()
        return model.theta.detach().clone()

    coupled = start("adam_l2")
    decoupled = start("adamw")
    assert torch.allclose(decoupled, torch.tensor([0.99, 9.9]), atol=1e-6)
    # Adam+L2 normalizes the gradient and has a different relative shrinkage.
    assert torch.allclose(coupled, torch.tensor([0.9, 9.9]), atol=1e-5)
    assert not torch.allclose(coupled, decoupled)


def test_adam_and_adamw_match_when_decay_is_zero():
    seed_everything(25)
    first = nn.Sequential(nn.Linear(3, 5), nn.Tanh(), nn.Linear(5, 2))
    second = copy.deepcopy(first)
    a = make_optimizer(first, "adam", lr=0.002, decay=0)
    b = make_optimizer(second, "adamw", lr=0.002, decay=0)
    x = torch.randn(7, 3)
    target = torch.randn(7, 2)
    for _ in range(5):
        for model, optimizer in ((first, a), (second, b)):
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(model(x), target).backward()
            optimizer.step()
    for x, y in zip(first.parameters(), second.parameters()):
        assert torch.allclose(x, y, atol=1e-7)


def test_identical_decay_parameter_mask():
    model = MoonMLP()
    groups = parameter_groups(model, 0.1)
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == pytest.approx(0.1)
    assert groups[1]["weight_decay"] == 0
    assert all(param.ndim >= 2 for param in groups[0]["params"])
    assert all(param.ndim < 2 for param in groups[1]["params"])


def test_moons_split_reproducible():
    d1 = _moon_data(42, torch.device("cpu"))
    d2 = _moon_data(42, torch.device("cpu"))
    assert [len(group[0]) for group in d1] == [160, 200, 2000]
    for (x1, y1), (x2, y2) in zip(d1, d2):
        assert torch.equal(x1, x2)
        assert torch.equal(y1, y2)
