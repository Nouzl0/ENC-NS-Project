```{raw} typst
#set page(margin: auto)
```

# **Implementation**

The experiment is implemented in Python 3.12 using PyTorch 2.8 as the core deep learning framework. All model definitions, training logic, data pipelines, and evaluation are contained in four source files under `src/`. Training, evaluation, and inference code is written directly against the PyTorch API The interactive demo is a separate Jupyter notebook in `example/`.

### Dependencies
Dependencies are managed with Poetry and pinned in `requirements.txt` for pip compatibility:

| Package | Role |
|---|---|
| `torch`, `torchvision` | Model training, transforms, DataLoaders, MNIST download |
| `numpy`, `matplotlib` | Metrics computation and report plots |
| `typer`, `rich` | CLI interface and formatted terminal output |
| `ipycanvas`, `ipywidgets` | Interactive drawing widget in the demo notebook |
| `tqdm` | Per-epoch and per-batch progress bars |
| `Pillow` | Image loading for the `predict` CLI command |


### Installation
Clone the repository and install dependencies using either Poetry or pip:

```bash
# Poetry (recommended)
$ poetry install

# pip
$ pip install -r requirements.txt
```


### Project Structure
The codebase is intentionally flat — four source files in `src/` cover the entire experiment pipeline. All commands in this document assume the working directory is the project root.

```
src/
├── data.py      — MNIST download, normalization strategies, DataLoader factory
├── models.py    — ModelFactory registry + all 5 architecture definitions
├── train.py     — Trainer (fit/evaluate/save) + Detector (inference)
├── main.py      — Typer CLI: train / report / predict
example/
├── canvas.py    — Jupyter drawing widget (ipycanvas + PyTorch inference)
└── demo.ipynb   — Interactive digit prediction notebook
models/          — Saved checkpoints (.pt) and metrics (.json) per run
output/          — Generated plots and reports
docs/            — MyST documentation source — the document you are reading
```

```{raw} typst
#pagebreak()
```


## **Data Preparation**

All data handling lives in `src/data.py`. The `MNISTData` class downloads the dataset on first run, applies the selected normalization strategy, splits the training set into training and validation subsets, and returns three ready-to-use DataLoaders. The same class is used for all 25 runs — only the normalization config changes between them, keeping everything else identical.

### Splitting

The official MNIST training set (60 000 samples) is split into training and validation subsets. MNISTData computes the split size dynamically from a configurable parameter — defaulting to 20%.

```python
n_val = int(val_split * len(full_train))   # 0.20 × 60 000 = 12 000
n_train = len(full_train) - n_val          # 48 000
random_split(full_train, [n_train, n_val], generator=torch.Generator().manual_seed(42))
```

The fixed seed makes the split identical across all 25 runs — every model and normalization combination trains and validates on exactly the same samples. The official test set (10 000 samples) is kept entirely separate — it is never seen during training or validation and is only used once per run for final evaluation.

### Normalization

Each normalization strategy is defined as a `(train_transform, eval_transform)` pair. For all static strategies the two are identical. For `augmented` they differ — geometric perturbations apply only during training; validation and test always use standardization to ensure fair evaluation:

```python
"none":        ToTensor() → ×255           # raw pixels, negative reference
"minmax":      ToTensor()                  # [0, 1]
"standardize": ToTensor() → Normalize(0.1307, 0.3081)
"symmetric":   ToTensor() → Normalize(0.5, 0.5)
"augmented":   RandomAffine(10°, ±10%, 90–110%) → Normalize(0.1307, 0.3081)
               eval: same as standardize
```

The validation subset is explicitly constructed with the eval transform regardless of what the training transform is — this is handled in `MNISTData._loaders()`.

### DataLoaders

Three DataLoaders are constructed per run — one for each split. The training loader shuffles samples each epoch; validation and test do not:

```python
DataLoader(train_ds, batch_size=512, shuffle=True)
DataLoader(val_ds,   batch_size=256, shuffle=False)
DataLoader(test_ds,  batch_size=256, shuffle=False)
```

The training batch is larger (512) to reduce the number of gradient updates per epoch and speed up training. Validation and test use 256 — no gradient storage is needed so a smaller batch is sufficient.

```{raw} typst
#pagebreak()
```


## **Model Architectures**

All five models are defined in `src/models.py` registered in `ModelFactory` — a simple decorator-based registry that maps a name string to a constructor. Calling `ModelFactory.create("DeepCNN")` returns a fresh instance without the training code needing to import it directly. Every model outputs 10 logits, one per digit class.

### LinearDNN — Baseline

The simplest possible architecture: one hidden layer, no convolutions, no regularization. Every pixel is treated as an independent feature — spatial relationships between neighboring pixels are completely ignored. Its role is to establish a lower bound on what a non-spatial model can achieve.

```python
Flatten(784) → Linear(784→128) → ReLU → Linear(128→10)
```

### DeepDNN — Deep Dense Network

Adds depth and regularization to the dense baseline. Three hidden layers with `BatchNorm` after each projection and `Dropout` between layers — this tests whether increased capacity and regularization can close the gap to convolutional models while still discarding spatial structure.

```python
Flatten(784)
  → Linear(784→512) → BatchNorm1d → ReLU → Dropout(0.3)
  → Linear(512→256) → BatchNorm1d → ReLU → Dropout(0.3)
  → Linear(256→128) → BatchNorm1d → ReLU
  → Linear(128→10)
```

### SimpleCNN — Convolutional Baseline
`
Minimal convolutional model — two conv blocks with no `BatchNorm` or `Dropout`. Introduces spatial awareness through learned filters while keeping the architecture as bare as possible, establishing the lower bound for CNN performance in this comparison.

```python
Conv2d(1→32, 3×3, pad=1) → ReLU → MaxPool2d(2)   # 28×28 → 14×14
Conv2d(32→64, 3×3, pad=1) → ReLU → MaxPool2d(2)  # 14×14 → 7×7
AdaptiveAvgPool2d(1) → Flatten(64)
  → Linear(64→128) → ReLU → Linear(128→10)
```

### DeepCNN — Deep Convolutional Network

Extends SimpleCNN with a third convolutional block, `BatchNorm` after each convolution, and `Dropout` in the classifier head. VGG-inspired — deeper feature extraction with regularization at every stage.

```python
Conv2d(1→32,   pad=1) → BatchNorm2d → ReLU → MaxPool2d(2)   # 28×28 → 14×14
Conv2d(32→64,  pad=1) → BatchNorm2d → ReLU → MaxPool2d(2)   # 14×14 → 7×7
Conv2d(64→128, pad=1) → BatchNorm2d → ReLU                  # 7×7, no pool
AdaptiveAvgPool2d(1) → Flatten(128)
  → Linear(128→256) → ReLU → Dropout(0.5)
  → Linear(256→128) → ReLU
  → Linear(128→10)
```

```{raw} typst
#pagebreak()
```

### MiniResNet — Residual Network

The most complex architecture. Residual (skip) connections let the network learn corrections to the identity rather than a full mapping, giving gradients a direct path through the network and preventing them from vanishing in deeper layers.

```python
Stem:   Conv2d(1→32, pad=1) → BatchNorm → ReLU             # 28×28

Layer1: ResBlock(32) → MaxPool2d(2)                         # 28×28 → 14×14
Layer2: Conv2d(32→64, pad=1) → BatchNorm → ReLU
        ResBlock(64) → MaxPool2d(2)                         # 14×14 → 7×7

Head:   AdaptiveAvgPool2d(1) → Flatten(64) → Linear(64→10)
```

Each `ResBlock` is two conv layers with a skip connection: `Conv→BN→ReLU→Conv→BN`, `output = F(x) + x → ReLU`. The head uses Global Average Pooling to collapse the spatial dimensions to one value per channel before classification — this replaces a large dense layer and keeps parameter count low compared to DeepCNN despite the additional residual blocks.

---


## **Training Pipeline**

All training logic lives in `src/train.py`. The `Trainer` class wraps a model, device, and hyperparameters, exposes a `fit()` method that runs the full training loop, and returns a `History` object with per-epoch loss and accuracy for both training and validation sets. Hyperparameters are fixed across all 25 runs — `EPOCHS = 15`, `LR = 1e-3`, `BATCH = 512`.

### Optimizer and Scheduler

```python
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
criterion = CrossEntropyLoss()
```

`CosineAnnealingLR` decays the learning rate smoothly from `lr` to `0` over the full 15-epoch run:

$$\eta_t = \frac{\eta_{\max}}{2}\left(1 + \cos\left(\frac{t\pi}{T_{\max}}\right)\right)$$

This avoids abrupt step-decay drops — the model converges gradually into the loss minimum rather than oscillating around it. `weight_decay=1e-4` adds L2 regularization directly through Adam.

### Mixed Precision and Compilation

On CUDA, the model is compiled at initialization with `torch.compile`, which fuses operations into optimized kernels via TorchInductor. During each forward pass, `bfloat16` mixed precision is enabled:

```python
with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
    logits = model(xb)
    loss = criterion(logits, yb)
```

`bfloat16` halves memory bandwidth while preserving the dynamic range of `float32`. On CPU autocast is disabled and the model runs in eager mode at full precision.

### Metrics Computed After Each Run

| Metric | Description |
|---|---|
| `test_acc` / `test_loss` | Final evaluation on the held-out test set |
| `per_class_acc` | Per-digit accuracy from the confusion matrix diagonal |
| `precision` / `recall` / `f1` | Per-class and macro/weighted averages |
| `mcc` | Multiclass Matthews Correlation Coefficient |
| `params` | Total trainable parameter count |
| `train_time_s` | Wall-clock training duration in seconds |
| `inference_ms_per_sample` | Average forward-pass time per sample on the test set |
| `ckpt_size_kb` | Checkpoint file size in kilobytes |

---


## **CLI**

The experiment workflow is exposed through a CLI in src/main.py. Run from the project root:

```bash
python src/main.py <command>   # module (installed with pip/poetry)
mnist-exp <command>            # installed entry point (only with poetry)
```

### train

Trains one or more model × normalization combinations and saves results to models/:

```bash
python src/main.py train --model SimpleCNN --norm augmented       # single run
python src/main.py train -m DeepCNN -m MiniResNet -n standardize  # 2×1 grid
python src/main.py train                                          # all 25
```

Hyperparameters can be overridden: `--epochs`, `--lr`, `--batch`. Each run saves `{Model}_{norm}.pt` (weights) and `{Model}_{norm}.json` (metrics) to `models/`.

### report

Reads saved JSON files and generates reports — no re-training required:

```bash
python src/main.py report all                  # ranked tables + heatmaps
python src/main.py report SimpleCNN_augmented  # single-model breakdown
```

### Predict command

Runs inference on a single image and prints per-digit probabilities:

```bash
python src/main.py predict digit.png  # best saved model
python src/main.py predict digit.png --model MiniResNet_augmented
```

### Default (no command)

Running python `src/main.py` without a subcommand trains all 25 combinations in sequence and then generates a full comparison report — equivalent to `train` followed by `report all`.

```{raw} typst
#pagebreak()
```

### Interactive Demo
The `example/demo.ipynb` notebook provides an interactive drawing widget implemented in `example/canvas.py`. Draw a digit in the browser and see live predictions from the best saved model.

```{figure} figures/example.png
:name: fig-demo
:width: 60%
Interactive digit recognition demo — digit 3 drawn on the canvas.
```

---
