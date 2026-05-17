from __future__ import annotations
from typing import Callable

import torch
import torch.nn as nn

NUM_CLASSES = 10  # MNIST digit classes
FLAT_DIM = 784  # 28×28 flattened


class ModelFactory:
    _registry: dict[str, Callable[[], nn.Module]] = {}

    @classmethod
    def register(
        cls, name: str
    ) -> Callable[[Callable[[], nn.Module]], Callable[[], nn.Module]]:
        def decorator(fn: Callable[[], nn.Module]):
            cls._registry[name] = fn
            return fn

        return decorator

    @classmethod
    def create(cls, name: str) -> nn.Module:
        if name not in cls._registry:
            raise KeyError(f"Unknown model: {name}")
        return cls._registry[name]()

    @classmethod
    def names(cls) -> list[str]:
        return list(cls._registry)


@ModelFactory.register("LinearDNN")
class LinearDNN(nn.Module):
    """Fully-connected baseline: one hidden layer, no convolutions."""

    IN_FEATURES = FLAT_DIM
    HIDDEN = 128
    OUT_FEATURES = NUM_CLASSES

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),  # (B, 1, 28, 28) → (B, 784)
            nn.Linear(self.IN_FEATURES, self.HIDDEN),  # learned projection
            nn.ReLU(),  # non-linearity
            nn.Linear(self.HIDDEN, self.OUT_FEATURES),  # → class logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@ModelFactory.register("DeepDNN")
class DeepDNN(nn.Module):
    """Deep fully-connected network with BatchNorm and Dropout for regularization."""

    IN_FEATURES = FLAT_DIM
    HIDDEN = (512, 256, 128)
    OUT_FEATURES = NUM_CLASSES
    DROPOUT = 0.3

    def __init__(self):
        super().__init__()
        h1, h2, h3 = self.HIDDEN

        self.net = nn.Sequential(
            nn.Flatten(),  # (B, 1, 28, 28) → (B, 784)
            nn.Linear(self.IN_FEATURES, h1),  # learned projection
            nn.BatchNorm1d(h1),  # stabilize activations
            nn.ReLU(),
            nn.Dropout(self.DROPOUT),  # randomly drop neurons
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(self.DROPOUT),
            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.ReLU(),
            nn.Linear(h3, self.OUT_FEATURES),  # → class logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@ModelFactory.register("SimpleCNN")
class SimpleCNN(nn.Module):
    """Two conv blocks without BatchNorm — intentionally simple CNN baseline."""

    CHANNELS = (1, 32, 64)  # input → conv1 → conv2
    HIDDEN = 128
    OUT_FEATURES = NUM_CLASSES  # one score per MNIST digit class

    def __init__(self):
        super().__init__()
        c_in, c1, c2 = self.CHANNELS

        self.features = nn.Sequential(
            nn.Conv2d(c_in, c1, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c2, self.HIDDEN),  # learned projection
            nn.ReLU(),
            nn.Linear(self.HIDDEN, self.OUT_FEATURES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


@ModelFactory.register("DeepCNN")
class DeepCNN(nn.Module):
    """Three conv blocks with BatchNorm + Dropout — stronger CNN."""

    CHANNELS = (1, 32, 64, 128)  # input → conv1 → conv2 → conv3
    HIDDEN = (256, 128)
    OUT_FEATURES = NUM_CLASSES  # one score per MNIST digit class
    DROPOUT = 0.5

    def __init__(self):
        super().__init__()
        c_in, c1, c2, c3 = self.CHANNELS
        h1, h2 = self.HIDDEN

        self.features = nn.Sequential(
            nn.Conv2d(c_in, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B,1,28,28) → (B,32,14,14)
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),
            nn.MaxPool2d(2),  # → (B,64,7,7)
            nn.Conv2d(c2, c3, 3, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(),  # → (B,128,7,7)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c3, h1),
            nn.ReLU(),
            nn.Dropout(self.DROPOUT),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, self.OUT_FEATURES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


@ModelFactory.register("MiniResNet")
class MiniResNet(nn.Module):
    """ResNet-inspired architecture — skip connections prevent vanishing gradients."""

    CHANNELS = (1, 32, 64)  # input → stem → layer2
    OUT_FEATURES = NUM_CLASSES  # one score per MNIST digit class

    class _ResBlock(nn.Module):
        """Two conv layers with a skip connection: output = F(x) + x."""

        def __init__(self, ch):
            super().__init__()

            self.block = nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(),  # conv + norm
                nn.Conv2d(ch, ch, 3, padding=1),
                nn.BatchNorm2d(ch),  # conv + norm (no ReLU — applied after skip)
            )
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.relu(
                self.block(x) + x
            )  # skip connection: adds input back before activation

    def __init__(self):
        super().__init__()
        c_in, c1, c2 = self.CHANNELS

        self.stem = nn.Sequential(
            nn.Conv2d(c_in, c1, 3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(),  # (B,1,28,28) → (B,32,28,28)
        )

        self.layer1 = nn.Sequential(
            self._ResBlock(c1),
            nn.MaxPool2d(2),  # → (B,32,14,14)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(),  # → (B,64,14,14)
            self._ResBlock(c2),
            nn.MaxPool2d(2),  # → (B,64,7,7)
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # → (B,64,1,1)  global average pool
            nn.Flatten(),  # → (B,64)
            nn.Linear(c2, self.OUT_FEATURES),  # → class logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.head(x)
