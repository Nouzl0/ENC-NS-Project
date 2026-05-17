from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        *,
        lr: float = 1e-3,
        epochs: int = 15,
        weight_decay: float = 1e-4,
    ) -> None:
        can_compile = device.type == "cuda"
        self._model = (
            torch.compile(model.to(device)) if can_compile else model.to(device)
        )
        self._device = device
        self._lr = lr
        self._epochs = epochs
        self._weight_decay = weight_decay

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> History:
        self._criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self._lr, weight_decay=self._weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self._epochs
        )
        history = History()

        bar = tqdm(range(1, self._epochs + 1), desc="Training", unit="epoch")
        for _ in bar:
            tr_loss, tr_acc = self._train_epoch(
                train_loader, self._criterion, optimizer
            )
            va_loss, va_acc = self._eval_epoch(val_loader, self._criterion)
            scheduler.step()

            history.train_loss.append(tr_loss)
            history.val_loss.append(va_loss)
            history.train_acc.append(tr_acc)
            history.val_acc.append(va_acc)

            bar.set_postfix(
                tr_loss=f"{tr_loss:.4f}",
                tr_acc=f"{tr_acc * 100:.1f}%",
                va_loss=f"{va_loss:.4f}",
                va_acc=f"{va_acc * 100:.1f}%",
            )

        return history

    def save(self, model_name: str, path: str | Path = "models") -> Path:
        out = Path(path) / f"{model_name}.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_name": model_name, "state_dict": self._model.state_dict()}, out
        )
        return out

    def evaluate(self, loader: DataLoader) -> tuple[float, float]:
        return self._eval_epoch(loader, self._criterion)

    def _train_epoch(
        self, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer
    ) -> tuple[float, float]:
        self._model.train()
        total_loss = torch.tensor(0.0, device=self._device)
        correct = torch.tensor(0, device=self._device)
        total = 0
        for xb, yb in tqdm(loader, desc="  train", leave=False):
            xb, yb = xb.to(self._device), yb.to(self._device)

            with torch.autocast(
                self._device.type,
                dtype=torch.bfloat16,
                enabled=self._device.type == "cuda",
            ):
                logits = self._model(xb)
                loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach() * len(yb)
            correct += (logits.detach().argmax(1) == yb).sum()
            total += len(yb)
        return (total_loss / total).item(), (correct / total).item()

    def _eval_epoch(
        self, loader: DataLoader, criterion: nn.Module
    ) -> tuple[float, float]:
        self._model.eval()
        total_loss = torch.tensor(0.0, device=self._device)
        correct = torch.tensor(0, device=self._device)
        total = 0
        with torch.inference_mode():
            for xb, yb in tqdm(loader, desc="  val  ", leave=False):
                xb, yb = xb.to(self._device), yb.to(self._device)
                with torch.autocast(
                    self._device.type,
                    dtype=torch.bfloat16,
                    enabled=self._device.type == "cuda",
                ):
                    logits = self._model(xb)
                    total_loss += criterion(logits, yb) * len(yb)
                correct += (logits.argmax(1) == yb).sum()
                total += len(yb)
        return (total_loss / total).item(), (correct / total).item()


_MNIST_CLASSES = [str(i) for i in range(10)]

_DETECT_TRANSFORM = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)


class Detector:
    """Loads a saved Trainer checkpoint and classifies MNIST digit images."""

    def __init__(
        self, checkpoint: str | Path, device: str | torch.device = "cpu"
    ) -> None:
        from models import ModelFactory

        self._device = torch.device(device)
        data = torch.load(checkpoint, map_location=self._device, weights_only=True)
        model = ModelFactory.create(data["model_name"])
        model.load_state_dict(data["state_dict"])
        self._model = model.to(self._device).eval()

    def predict(self, image: str | Path | Image.Image) -> tuple[str, float]:
        """Return (predicted_label, confidence) for a single image."""
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        tensor = _DETECT_TRANSFORM(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            probs = F.softmax(self._model(tensor), dim=1)[0]
        idx = probs.argmax().item()
        return _MNIST_CLASSES[idx], probs[idx].item()

    def predict_batch(
        self, images: list[str | Path | Image.Image]
    ) -> list[tuple[str, float]]:
        return [self.predict(img) for img in images]

    def probs(self, image: str | Path | Image.Image) -> list[float]:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        tensor = _DETECT_TRANSFORM(image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            return F.softmax(self._model(tensor), dim=1)[0].tolist()
