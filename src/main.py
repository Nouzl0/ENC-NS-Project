from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Optional

import torch
import torch.nn.functional as F
import typer
from PIL import Image as PILImage

from .data import MNISTData, _CONFIGS
from .models import ModelFactory
from .train import Detector, Trainer, _DETECT_TRANSFORM

EPOCHS = 15
BATCH = 512
LR = 1e-3
VAL_SPLIT = 0.20
MODELS_DIR = Path("models")
OUTPUT_DIR = Path("output")

_MODEL_NAMES: list[str] = ModelFactory.names()
_NORM_NAMES: list[str] = list(_CONFIGS.keys())

app = typer.Typer(
    help=(
        "MNIST Neural Network Experiment Suite.\n\n"
        "Train and compare 5 model architectures (LinearDNN, DeepDNN, SimpleCNN, DeepCNN, MiniResNet) "
        "across 5 normalization strategies (none, minmax, standardize, symmetric, augmented).\n\n"
        "Run without a subcommand to train all 25 combinations and generate a full comparison report."
    ),
    invoke_without_command=True,
    add_completion=False,
)


def _setup_device(seed: int = 42) -> torch.device:
    match True:
        case _ if torch.backends.mps.is_available():
            device = torch.device("mps")
        case _ if torch.cuda.is_available():
            device = torch.device("cuda")
        case _:
            device = torch.device("cpu")
    torch.manual_seed(seed)
    typer.echo(f"Device: {device}  |  threads: {torch.get_num_threads()}")
    return device


def _confusion_matrix(model, loader, device) -> list[list[int]]:
    matrix = [[0] * 10 for _ in range(10)]
    model.eval()
    with torch.inference_mode():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1)
            for true, pred in zip(yb.tolist(), preds.tolist()):
                matrix[true][pred] += 1
    return matrix


def _class_metrics(matrix: list[list[int]]) -> dict:
    """Compute per-class precision, recall, F1 + macro/weighted averages and multiclass MCC."""
    n = len(matrix)
    support = [sum(matrix[i]) for i in range(n)]
    total = sum(support)
    col_sums = [sum(matrix[j][i] for j in range(n)) for i in range(n)]

    per_p, per_r, per_f1 = [], [], []
    for i in range(n):
        tp = matrix[i][i]
        fp = col_sums[i] - tp
        fn = support[i] - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_p.append(round(p, 6))
        per_r.append(round(r, 6))
        per_f1.append(round(f1, 6))

    macro_p = round(sum(per_p) / n, 6)
    macro_r = round(sum(per_r) / n, 6)
    macro_f1 = round(sum(per_f1) / n, 6)

    weighted_p = (
        round(sum(per_p[i] * support[i] for i in range(n)) / total, 6) if total else 0.0
    )
    weighted_r = (
        round(sum(per_r[i] * support[i] for i in range(n)) / total, 6) if total else 0.0
    )
    weighted_f1 = (
        round(sum(per_f1[i] * support[i] for i in range(n)) / total, 6)
        if total
        else 0.0
    )

    # Gorodkin multiclass MCC
    correct = sum(matrix[i][i] for i in range(n))
    num = total * correct - sum(support[i] * col_sums[i] for i in range(n))
    den_a = total * total - sum(s**2 for s in support)
    den_b = total * total - sum(c**2 for c in col_sums)
    mcc = round(num / (den_a * den_b) ** 0.5, 6) if (den_a * den_b) > 0 else 0.0

    return {
        "precision": {"per_class": per_p, "macro": macro_p, "weighted": weighted_p},
        "recall": {"per_class": per_r, "macro": macro_r, "weighted": weighted_r},
        "f1": {"per_class": per_f1, "macro": macro_f1, "weighted": weighted_f1},
        "mcc": mcc,
    }


def _inference_time_ms(model, loader, device) -> float:
    """Mean inference time in milliseconds per sample over the full loader."""
    model.eval()
    elapsed = 0.0
    n_samples = 0
    with torch.inference_mode():
        for xb, _ in loader:
            xb = xb.to(device)
            t0 = time.perf_counter()
            model(xb)
            elapsed += time.perf_counter() - t0
            n_samples += len(xb)
    return round(elapsed / n_samples * 1000, 4) if n_samples else 0.0


def _run_one(
    model_name: str, norm: str, device, *, epochs: int, lr: float, batch: int
) -> dict:
    data = MNISTData(batch=batch, val_split=VAL_SPLIT, device=device)
    loaders = getattr(data, norm)()

    raw_model = ModelFactory.create(model_name)
    params = sum(p.numel() for p in raw_model.parameters())

    trainer = Trainer(raw_model, device, lr=lr, epochs=epochs)

    t0 = time.perf_counter()
    history = trainer.fit(loaders.train, loaders.val)
    train_time_s = round(time.perf_counter() - t0, 2)

    test_loss, test_acc = trainer.evaluate(loaders.test)
    inference_ms = _inference_time_ms(trainer._model, loaders.test, device)

    conf_matrix = _confusion_matrix(trainer._model, loaders.test, device)
    metrics = _class_metrics(conf_matrix)
    per_class_acc = [
        round(conf_matrix[i][i] / max(sum(conf_matrix[i]), 1), 6) for i in range(10)
    ]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = trainer.save(f"{model_name}_{norm}", path=MODELS_DIR)
    ckpt_size_kb = round(ckpt.stat().st_size / 1024, 1)

    result = {
        "model": model_name,
        "norm": norm,
        "epochs": epochs,
        "lr": lr,
        "batch": batch,
        "params": params,
        "train_time_s": train_time_s,
        "inference_ms_per_sample": inference_ms,
        "ckpt_size_kb": ckpt_size_kb,
        "test_loss": round(test_loss, 6),
        "test_acc": round(test_acc, 6),
        "per_class_acc": per_class_acc,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "mcc": metrics["mcc"],
        "confusion_matrix": conf_matrix,
        "checkpoint": str(ckpt),
        "history": {
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
            "train_acc": history.train_acc,
            "val_acc": history.val_acc,
        },
    }
    (MODELS_DIR / f"{model_name}_{norm}.json").write_text(json.dumps(result, indent=2))
    return result


def _load_all_metrics() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(MODELS_DIR.glob("*.json"))]


def _report_single(data: dict) -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    name = f"{data['model']}_{data['norm']}"
    conf = np.array(data["confusion_matrix"])
    f1, precision, recall = data["f1"], data["precision"], data["recall"]

    # ── text summary ──────────────────────────────────────────────────────────
    typer.echo(f"\n{'═' * 66}")
    typer.echo(f"  {name}")
    typer.echo(f"{'═' * 66}")
    typer.echo(f"  Test acc:    {data['test_acc'] * 100:.2f}%")
    typer.echo(f"  Test loss:   {data['test_loss']:.4f}")
    typer.echo(f"  MCC:         {data['mcc']:.4f}")
    typer.echo(
        f"  Macro  F1 / P / R:    {f1['macro'] * 100:.2f}%  {precision['macro'] * 100:.2f}%  {recall['macro'] * 100:.2f}%"
    )
    typer.echo(
        f"  Weighted F1 / P / R:  {f1['weighted'] * 100:.2f}%  {precision['weighted'] * 100:.2f}%  {recall['weighted'] * 100:.2f}%"
    )
    typer.echo(f"  Params:      {data['params']:,}")
    typer.echo(f"  Train time:  {data['train_time_s']:.1f}s")
    typer.echo(f"  Inference:   {data['inference_ms_per_sample']:.3f} ms/sample")
    typer.echo(f"  Ckpt size:   {data['ckpt_size_kb']} KB")
    typer.echo(f"\n  {'Digit':<7} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Acc':>8}")
    typer.echo(f"  {'─' * 44}")
    for i in range(10):
        typer.echo(
            f"  {i:<7}"
            f" {precision['per_class'][i] * 100:>9.2f}%"
            f" {recall['per_class'][i] * 100:>7.2f}%"
            f" {f1['per_class'][i] * 100:>7.2f}%"
            f" {data['per_class_acc'][i] * 100:>7.2f}%"
        )
    typer.echo(f"{'═' * 66}\n")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(name, fontsize=13, fontweight="bold")

    # confusion matrix
    ax = axes[0]
    row_sums = conf.sum(axis=1, keepdims=True).clip(1)
    norm_conf = conf / row_sums
    im = ax.imshow(norm_conf, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    for i in range(10):
        for j in range(10):
            ax.text(
                j,
                i,
                str(conf[i, j]),
                ha="center",
                va="center",
                fontsize=7,
                color="white" if norm_conf[i, j] > 0.5 else "black",
            )
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # per-class precision / recall / F1 grouped bars
    ax = axes[1]
    x = list(range(10))
    w = 0.26
    all_vals = (
        [v * 100 for v in precision["per_class"]]
        + [v * 100 for v in recall["per_class"]]
        + [v * 100 for v in f1["per_class"]]
    )
    y_min = max(min(all_vals) - 1.5, 0)

    bars_p = ax.bar([v - w for v in x], [v * 100 for v in precision["per_class"]],
                    width=w, label="Precision", color="steelblue")
    bars_r = ax.bar(list(x), [v * 100 for v in recall["per_class"]],
                    width=w, label="Recall", color="tomato")
    bars_f = ax.bar([v + w for v in x], [v * 100 for v in f1["per_class"]],
                    width=w, label="F1", color="seagreen")
    ax.axhline(f1["weighted"] * 100, color="black", linestyle="--", linewidth=1,
               label=f"Weighted F1 {f1['weighted'] * 100:.1f}%")

    for bars in (bars_p, bars_r, bars_f):
        for bar in bars:
            h_val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h_val + 0.05,
                    f"{h_val:.1f}", ha="center", va="bottom", fontsize=5.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xlabel("Digit")
    ax.set_ylabel("%")
    ax.set_ylim(y_min, 101.5)
    ax.legend(fontsize=8)
    ax.set_title("Per-class Precision / Recall / F1")

    # learning curves
    ax = axes[2]
    h = data["history"]
    ep = range(1, len(h["train_acc"]) + 1)
    ax.plot(ep, [a * 100 for a in h["train_acc"]], label="Train acc", color="steelblue")
    ax.plot(
        ep,
        [a * 100 for a in h["val_acc"]],
        label="Val acc",
        color="tomato",
        linestyle="--",
    )
    ax.axhline(
        data["test_acc"] * 100,
        color="green",
        linestyle=":",
        linewidth=1.5,
        label=f"Test {data['test_acc'] * 100:.2f}%",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Learning Curves")

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / f"{name}_report.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    typer.echo(f"Plot saved → {plot_path}")
    plt.show()


def _report_all(runs: list[dict]) -> None:
    import numpy as np
    import matplotlib.pyplot as plt
    
    sorted_runs = sorted(runs, key=lambda r: r["test_acc"], reverse=True)

    # ── performance table ─────────────────────────────────────────────────────
    w = [4, 14, 13, 10, 10, 11, 8]
    sep = "─" * (sum(w) + 3 * len(w))
    typer.echo(f"\n{'═' * len(sep)}")
    typer.echo("  PERFORMANCE — ranked by Test Accuracy")
    typer.echo(f"{'═' * len(sep)}")
    typer.echo(
        f"  {'#':<{w[0]}} {'Model':<{w[1]}} {'Norm':<{w[2]}}"
        f" {'Test Acc':>{w[3]}} {'Macro F1':>{w[4]}} {'Wtd F1':>{w[5]}} {'MCC':>{w[6]}}"
    )
    typer.echo(sep)
    for rank, r in enumerate(sorted_runs, 1):
        typer.echo(
            f"  {rank:<{w[0]}} {r['model']:<{w[1]}} {r['norm']:<{w[2]}}"
            f" {r['test_acc'] * 100:>{w[3]}.2f}%"
            f" {r['f1']['macro'] * 100:>{w[4]}.2f}%"
            f" {r['f1']['weighted'] * 100:>{w[5]}.2f}%"
            f" {r['mcc']:>{w[6]}.4f}"
        )
    typer.echo(f"{'═' * len(sep)}\n")

    # ── efficiency table ──────────────────────────────────────────────────────
    we = [4, 14, 13, 10, 12, 12, 10]
    sep2 = "─" * (sum(we) + 3 * len(we))
    typer.echo(f"{'═' * len(sep2)}")
    typer.echo("  EFFICIENCY")
    typer.echo(f"{'═' * len(sep2)}")
    typer.echo(
        f"  {'#':<{we[0]}} {'Model':<{we[1]}} {'Norm':<{we[2]}}"
        f" {'Params':>{we[3]}} {'Train (s)':>{we[4]}} {'Infer (ms)':>{we[5]}} {'Size (KB)':>{we[6]}}"
    )
    typer.echo(sep2)
    for rank, r in enumerate(sorted_runs, 1):
        typer.echo(
            f"  {rank:<{we[0]}} {r['model']:<{we[1]}} {r['norm']:<{we[2]}}"
            f" {r['params']:>{we[3]},}"
            f" {r['train_time_s']:>{we[4]}.1f}"
            f" {r['inference_ms_per_sample']:>{we[5]}.3f}"
            f" {r['ckpt_size_kb']:>{we[6]}.1f}"
        )
    typer.echo(f"{'═' * len(sep2)}\n")

    if len(runs) < 2:
        typer.echo(
            "Only one model saved — skipping comparison plots. Train more models first."
        )
        return

    model_names = list(dict.fromkeys(r["model"] for r in runs))
    norm_names = list(dict.fromkeys(r["norm"] for r in runs))
    acc_map = {(r["model"], r["norm"]): r["test_acc"] for r in runs}
    wf1_map = {(r["model"], r["norm"]): r["f1"]["weighted"] for r in runs}
    mcc_map = {(r["model"], r["norm"]): r["mcc"] for r in runs}

    def _heat(src):
        return np.array(
            [[src.get((m, n), 0.0) for m in model_names] for n in norm_names]
        )

    heat_acc = _heat(acc_map) * 100
    heat_wf1 = _heat(wf1_map) * 100
    heat_mcc = _heat(mcc_map) * 100

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle("MNIST — All Models Comparison", fontsize=14, fontweight="bold")

    for ax, heat, title in zip(
        axes,
        [heat_acc, heat_wf1, heat_mcc],
        ["Test Accuracy (%)", "Weighted F1 (%)", "MCC (×100)"],
    ):
        im = ax.imshow(heat, cmap="RdYlGn", vmin=max(heat.min() - 1, 0), vmax=100)
        plt.colorbar(im, ax=ax, label=title)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=15, ha="right")
        ax.set_yticks(range(len(norm_names)))
        ax.set_yticklabels(norm_names)
        for i in range(len(norm_names)):
            for j in range(len(model_names)):
                ax.text(j, i, f"{heat[i, j]:.1f}", ha="center", va="center", fontsize=9)
        ax.set_title(title)

    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = OUTPUT_DIR / "comparison_report.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    typer.echo(f"Plot saved → {plot_path}")
    plt.show()


@app.callback()
def default(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is not None:
        return
    typer.echo("No command given — training all models and generating full report.\n")
    device = _setup_device()
    total = len(_MODEL_NAMES) * len(_NORM_NAMES)
    for i, m in enumerate(_MODEL_NAMES):
        for j, n in enumerate(_NORM_NAMES):
            idx = i * len(_NORM_NAMES) + j + 1
            typer.echo(f"\n{'─' * 60}")
            typer.echo(f"[{idx}/{total}]  {m}  ×  {n}")
            typer.echo("─" * 60)
            result = _run_one(m, n, device, epochs=EPOCHS, lr=LR, batch=BATCH)
            typer.echo(f"  → test acc: {result['test_acc'] * 100:.2f}%")
    _report_all(_load_all_metrics())


@app.command()
def train(
    models: Annotated[
        Optional[list[str]],
        typer.Option(
            "--model",
            "-m",
            help=(
                "Model architecture to train. "
                "Choices: LinearDNN, DeepDNN, SimpleCNN, DeepCNN, MiniResNet. "
            ),
        ),
    ] = None,
    norms: Annotated[
        Optional[list[str]],
        typer.Option(
            "--norm",
            "-n",
            help=(
                "Normalization strategy to use. "
                "Choices: none, minmax, standardize, symmetric, augmented. "
            ),
        ),
    ] = None,
    epochs: Annotated[int, typer.Option(help="Number of training epochs.")] = EPOCHS,
    lr: Annotated[float, typer.Option(help="Adam optimizer learning rate.")] = LR,
    batch: Annotated[
        int, typer.Option(help="Batch size for the training DataLoader.")
    ] = BATCH,
) -> None:
    """Train model × normalization combinations and save results to models/.

    Each --model × --norm pair is one training run. Specifying 2 models and 3 norms
    produces 6 runs. Checkpoints (.pt) and metrics (.json) are saved to models/.

    Examples:
      train                                          # all 25 combinations
      train --model SimpleCNN --norm augmented       # single run
      train -m DeepCNN -m MiniResNet -n standardize  # 2 models × 1 norm = 2 runs
    """
    model_list = models or _MODEL_NAMES
    norm_list = norms or _NORM_NAMES

    for m in model_list:
        if m not in _MODEL_NAMES:
            raise typer.BadParameter(
                f"Unknown model '{m}'. Choose from: {_MODEL_NAMES}"
            )
    for n in norm_list:
        if n not in _NORM_NAMES:
            raise typer.BadParameter(f"Unknown norm '{n}'. Choose from: {_NORM_NAMES}")

    device = _setup_device()
    total = len(model_list) * len(norm_list)
    for i, m in enumerate(model_list):
        for j, n in enumerate(norm_list):
            idx = i * len(norm_list) + j + 1
            typer.echo(f"\n{'─' * 60}")
            typer.echo(f"[{idx}/{total}]  {m}  ×  {n}")
            typer.echo("─" * 60)
            result = _run_one(m, n, device, epochs=epochs, lr=lr, batch=batch)
            typer.echo(f"  → test acc: {result['test_acc'] * 100:.2f}%")


@app.command()
def report(
    target: Annotated[
        str,
        typer.Argument(
            help=(
                "'all' — compare all saved models side by side (accuracy + F1 + MCC heatmaps, ranked tables). "
                "model_key — detailed report for one model, e.g. 'SimpleCNN_augmented' "
                "(confusion matrix, per-class precision/recall/F1, learning curves). "
            )
        ),
    ],
) -> None:
    """Generate evaluation reports from saved training results.

    Reads metrics from models/*.json — no re-training required.
    Plots are saved to output/.

    Examples:
      report all                  # full comparison across all saved models
      report SimpleCNN_augmented  # detailed breakdown for one model
    """
    if target == "all":
        runs = _load_all_metrics()
        if not runs:
            typer.echo(f"No saved models found in {MODELS_DIR}/. Run 'train' first.")
            raise typer.Exit(1)
        _report_all(runs)
    else:
        metrics_file = MODELS_DIR / f"{target}.json"
        if not metrics_file.exists():
            available = [p.stem for p in sorted(MODELS_DIR.glob("*.json"))]
            typer.echo(f"Model '{target}' not found. Available: {available}")
            raise typer.Exit(1)
        _report_single(json.loads(metrics_file.read_text()))


@app.command()
def predict(
    image: Annotated[
        Path,
        typer.Argument(help="Path to the image file containing a handwritten digit."),
    ],
    model: Annotated[
        Optional[str],
        typer.Option(
            "--model",
            "-m",
            help=(
                "Model key to use, e.g. 'SimpleCNN_augmented'. "
                "Must match a .pt file in models/. "
                "If omitted, the model with the highest saved test accuracy is used automatically."
            ),
        ),
    ] = None,
) -> None:
    """Predict the digit class in an image using a trained model.

    Prints the probability for each digit 0–9, with the predicted class marked.

    Examples:\n
      predict digit.png                              # uses best saved model\n
      predict digit.png --model SimpleCNN_augmented  # use a specific model
    """
    if model is None:
        jsons = list(MODELS_DIR.glob("*.json"))
        if not jsons:
            typer.echo(f"No saved models in {MODELS_DIR}/. Run 'train' first.")
            raise typer.Exit(1)
        best = max(jsons, key=lambda p: json.loads(p.read_text())["test_acc"])
        model = best.stem
        typer.echo(f"Using best model: {model}")

    ckpt = MODELS_DIR / f"{model}.pt"
    if not ckpt.exists():
        typer.echo(f"Checkpoint not found: {ckpt}")
        raise typer.Exit(1)

    detector = Detector(ckpt)
    tensor = _DETECT_TRANSFORM(PILImage.open(image)).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(detector._model(tensor), dim=1)[0].tolist()

    pred_idx = max(range(10), key=lambda i: probs[i])
    typer.echo(f"\nPredictions for: {image.name}  (model: {model})")
    typer.echo("─" * 30)
    for i, p in enumerate(probs):
        marker = "  ◄" if i == pred_idx else ""
        typer.echo(f"  {i}:  {p * 100:6.2f}%{marker}")
    typer.echo("─" * 30)


if __name__ == "__main__":
    app()
