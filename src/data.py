from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


class _ScaleTo255:
    """Converts [0, 1] tensor to [0, 255] — lambda can't be pickled by DataLoader workers."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * 255.0


_STANDARDIZE = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

# Each entry is (train_transform, eval_transform).
# For plain normalizations both are identical; for augmented variants they differ.
_CONFIGS: dict[str, tuple] = {
    "none": (
        transforms.Compose([transforms.ToTensor(), _ScaleTo255()]),
        transforms.Compose([transforms.ToTensor(), _ScaleTo255()]),
    ),
    "minmax": (
        transforms.ToTensor(),
        transforms.ToTensor(),
    ),
    "standardize": (
        _STANDARDIZE,
        _STANDARDIZE,
    ),
    "symmetric": (
        transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
        transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        ),
    ),
    "augmented": (
        transforms.Compose(
            [
                transforms.RandomAffine(
                    degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        ),
        _STANDARDIZE,  # val/test: no augmentation, same stats as standardize
    ),
}


class MNISTData:
    def __init__(
        self,
        root: str = "data",
        *,
        batch: int = 64,
        val_split: float = 0.1,
        device: torch.device | None = None,
    ) -> None:
        self._root = root
        self._batch = batch
        self._val_split = val_split
        self._device = device or torch.device("cpu")
        datasets.MNIST(root, train=True, download=True)
        datasets.MNIST(root, train=False, download=True)

    def _loaders(self, train_transform, eval_transform) -> DataLoaders:
        full_train = datasets.MNIST(
            self._root, train=True, download=False, transform=train_transform
        )
        test_ds = datasets.MNIST(
            self._root, train=False, download=False, transform=eval_transform
        )

        n_val = int(self._val_split * len(full_train))
        train_ds, val_ds = random_split(
            full_train,
            [len(full_train) - n_val, n_val],
            generator=torch.Generator().manual_seed(42),
        )
        # val subset needs the eval transform, not the train (augmented) one
        val_ds.dataset = datasets.MNIST(
            self._root, train=True, download=False, transform=eval_transform
        )

        n_workers = 0 if os.name == "nt" else min(os.cpu_count() or 2, 8)
        kw = dict(
            num_workers=n_workers,
            pin_memory=self._device.type == "cuda",
            persistent_workers=n_workers > 0,
        )
        return DataLoaders(
            train=DataLoader(train_ds, batch_size=self._batch, shuffle=True, **kw),
            val=DataLoader(val_ds, batch_size=256, shuffle=False, **kw),
            test=DataLoader(test_ds, batch_size=256, shuffle=False, **kw),
        )

    def none(self) -> DataLoaders:
        return self._loaders(*_CONFIGS["none"])

    def minmax(self) -> DataLoaders:
        return self._loaders(*_CONFIGS["minmax"])

    def standardize(self) -> DataLoaders:
        return self._loaders(*_CONFIGS["standardize"])

    def symmetric(self) -> DataLoaders:
        return self._loaders(*_CONFIGS["symmetric"])

    def augmented(self) -> DataLoaders:
        return self._loaders(*_CONFIGS["augmented"])

    def all(self) -> dict[str, DataLoaders]:
        return {name: self._loaders(*cfg) for name, cfg in _CONFIGS.items()}

    def names(self) -> list[str]:
        return list(_CONFIGS)
