# ENC-NS_Project

MNIST digit classification study comparing five neural network architectures across five input normalization strategies — 25 controlled runs under identical hyperparameters. Built with PyTorch and a custom CLI.

## Results

| Model | Params | Best Acc | Best Norm |
|---|---:|---:|---|
| MiniResNet | 112,394 | 99.58% | augmented |
| DeepCNN | 160,330 | 99.02% | augmented |
| DeepDNN | 569,226 | 98.54% | augmented |
| LinearDNN | 101,770 | 97.46% | standardize |
| SimpleCNN | 28,426 | 96.15% | none |

## Installation

```bash
# Poetry (recommended)
poetry install

# pip
pip install -r requirements.txt
```

## Usage

```bash
# Train a single model
python -m src.main train --model miniresnet --norm augmented

# Train all 25 combinations
python -m src.main train --all

# Generate plots and metrics report
python -m src.main report

# Run inference on an image
python -m src.main predict path/to/image.png --model miniresnet --norm augmented
```

Available models: `lineardnn`, `deepdnn`, `simplecnn`, `deepcnn`, `miniresnet`

Available normalizations: `none`, `minmax`, `standardize`, `symmetric`, `augmented`

## Project Structure

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
docs/            — MyST documentation source
```

## Demo

Open `example/demo.ipynb` in Jupyter to draw digits on a canvas and get live predictions from the best model (MiniResNet augmented).

## Authors

Nikolas Nosál, Ondřej Studničný, Jakub Vaněk — Mendel University in Brno, Faculty of Business and Economics