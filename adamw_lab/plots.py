"""Publication-friendly PNGs plus standalone Plotly HTML for GitHub artifacts."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from loguru import logger


def _export_interactive(fig: go.Figure, path: Path) -> None:
    fig.update_layout(template="plotly_white", hovermode="x unified")
    fig.write_html(str(path), include_plotlyjs=True, full_html=True)
    logger.info("interactive chart -> {}", path)


def plot_curves(out: Path, rows: list[dict], metrics: list[str]) -> None:
    """Mean ± standard deviation over paired random seeds."""
    fig, axes = plt.subplots(
        len(metrics), 1, figsize=(10, 3.4 * len(metrics)), squeeze=False
    )
    interactive = go.Figure()
    methods = sorted({row["method"] for row in rows})
    for axis, metric in zip(axes[:, 0], metrics):
        for method in methods:
            by_step = defaultdict(list)
            for row in rows:
                if row["method"] == method and metric in row:
                    by_step[int(row["step"])].append(float(row[metric]))
            xs = sorted(by_step)
            if not xs:
                continue
            means = np.asarray([np.mean(by_step[x]) for x in xs])
            stds = np.asarray([np.std(by_step[x]) for x in xs])
            axis.plot(xs, means, label=method, linewidth=2)
            axis.fill_between(xs, means - stds, means + stds, alpha=0.14)
            interactive.add_trace(go.Scatter(
                x=xs, y=means, mode="lines+markers", name=method,
                legendgroup=method, visible=(metric == metrics[0]),
                customdata=[metric] * len(xs), hovertemplate=(
                    "step=%{x}<br>mean=%{y:.5f}<br>metric=%{customdata}<extra></extra>"
                )
            ))
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.set_xlabel("Optimizer step / epoch")
        axis.grid(alpha=0.20)
        axis.legend()
    fig.suptitle("Adam + L2 vs AdamW | identical seeds and task loss", fontsize=14)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Interactive HTML: all metrics available using dropdown; one metric at a time.
    traces_per_metric = len(methods)
    # Every metric has one trace per method when all metrics are available.
    buttons = []
    for metric in metrics:
        visible = [trace.customdata[0] == metric for trace in interactive.data]
        buttons.append({
            "label": metric,
            "method": "update",
            "args": [{"visible": visible}, {"title": f"Training dynamics — {metric}",
                                            "yaxis": {"title": metric}}],
        })
    interactive.update_layout(
        title=f"Training dynamics — {metrics[0]}", xaxis_title="Step / epoch",
        yaxis_title=metrics[0],
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 0, "y": 1.18}],
    )
    _export_interactive(interactive, out / "training_curves.html")


def plot_quadratic(out: Path, traces: dict, *, start: tuple, decay: float) -> None:
    x = np.linspace(-3, 3.4, 220)
    y = np.linspace(-0.4, 3.3, 220)
    xx, yy = np.meshgrid(x, y)
    zz = 0.5 * ((xx - 2) ** 2 + 20 * (yy - 1) ** 2)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contour(xx, yy, zz, levels=np.geomspace(0.01, 150, 22), cmap="Greys", alpha=0.7)
    interactive = go.Figure(data=[
        go.Contour(x=x, y=y, z=zz, contours={"coloring": "lines"},
                   showscale=False, name="task loss")
    ])
    for method, path in traces.items():
        ax.plot(path[:, 0], path[:, 1], label=method, marker=".", markevery=max(1, len(path)//12))
        interactive.add_trace(go.Scatter(
            x=path[:, 0], y=path[:, 1], mode="lines+markers", name=method,
            hovertemplate="w1=%{x:.4f}<br>w2=%{y:.4f}<extra></extra>"
        ))
    ax.scatter([start[0]], [start[1]], marker="s", color="black", label="start")
    ax.scatter([2], [1], marker="*", s=170, color="green", label="task optimum")
    ax.set(xlabel="w1", ylabel="w2",
           title=f"Quadratic optimizer trajectories | coefficient={decay}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "quadratic_trajectories.png", dpi=175)
    plt.close(fig)
    interactive.update_layout(title="Quadratic optimizer trajectories", xaxis_title="w1",
                              yaxis_title="w2")
    _export_interactive(interactive, out / "quadratic_trajectories.html")


def plot_boundary(out: Path, data: tuple, model, method: str) -> None:
    import torch
    (train_x, train_y), (val_x, val_y), _ = data
    x_np = train_x.detach().cpu().numpy()
    lo = x_np.min(axis=0) - 0.8
    hi = x_np.max(axis=0) + 0.8
    gx, gy = np.meshgrid(
        np.linspace(lo[0], hi[0], 220), np.linspace(lo[1], hi[1], 220)
    )
    points = torch.tensor(
        np.column_stack([gx.ravel(), gy.ravel()]), dtype=torch.float32,
        device=train_x.device,
    )
    model.eval()
    with torch.inference_mode():
        values = torch.sigmoid(model(points)).reshape(gx.shape).cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fill = ax.contourf(gx, gy, values, levels=np.linspace(0, 1, 21), cmap="RdBu", alpha=0.65)
    ax.contour(gx, gy, values, levels=[0.5], colors="black", linewidths=1.6)
    ax.scatter(x_np[:, 0], x_np[:, 1], c=train_y.cpu().numpy(),
               cmap="bwr", s=16, edgecolors="black", linewidths=0.2)
    ax.set(title=f"{method} | Two Moons decision boundary", xlabel="feature 1", ylabel="feature 2")
    fig.colorbar(fill, ax=ax, label="Predicted P(class=1)")
    fig.tight_layout()
    fig.savefig(out / f"boundary_{method}.png", dpi=170)
    plt.close(fig)

    # Interactive heatmap and training points.
    interactive = go.Figure(data=[
        go.Heatmap(x=gx[0], y=gy[:, 0], z=values, colorscale="RdBu",
                   zmin=0, zmax=1, colorbar={"title": "P(y=1)"})
    ])
    interactive.add_trace(go.Scatter(
        x=x_np[:, 0], y=x_np[:, 1], mode="markers", name="train",
        marker={"color": train_y.cpu().numpy(), "colorscale": "RdBu",
                "size": 5, "line": {"width": 0.5, "color": "black"}}
    ))
    interactive.update_layout(title=f"{method} — Two Moons", xaxis_title="x1", yaxis_title="x2")
    _export_interactive(interactive, out / f"boundary_{method}.html")


def plot_sweep(out: Path, rows: list[dict]) -> None:
    methods = sorted({r["method"] for r in rows})
    lrs = sorted({r["lr"] for r in rows})
    decays = sorted({r["decay"] for r in rows})
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4.5),
                             squeeze=False)
    for method, ax in zip(methods, axes[0]):
        grid = np.full((len(lrs), len(decays)), np.nan)
        for i, lr in enumerate(lrs):
            for j, decay in enumerate(decays):
                vals = [r["val_loss"] for r in rows
                        if r["method"] == method and r["lr"] == lr and r["decay"] == decay]
                if vals:
                    grid[i, j] = np.mean(vals)
        img = ax.imshow(grid, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(decays)), [str(x) for x in decays], rotation=40)
        ax.set_yticks(range(len(lrs)), [str(x) for x in lrs])
        ax.set(xlabel="L2 / weight decay coefficient", ylabel="Learning rate",
               title=method + " — mean validation BCE")
        for i in range(len(lrs)):
            for j in range(len(decays)):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center",
                            color="white" if grid[i,j] < np.nanmean(grid) else "black",
                            fontsize=8)
        fig.colorbar(img, ax=ax, label="Validation BCE")
        interactive = go.Figure(data=go.Heatmap(
            x=[str(x) for x in decays], y=[str(x) for x in lrs], z=grid,
            colorscale="Viridis", hovertemplate="wd=%{x}<br>lr=%{y}<br>BCE=%{z:.4f}<extra></extra>"
        ))
        interactive.update_layout(title=method, xaxis_title="Decay", yaxis_title="LR")
        _export_interactive(interactive, out / f"sweep_{method}.html")
    fig.tight_layout()
    fig.savefig(out / "sweep_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
