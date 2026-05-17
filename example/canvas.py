import io
import math

import numpy as np
import ipywidgets as widgets
from ipycanvas import Canvas, hold_canvas
from PIL import Image
from IPython.display import display

_LABELS = [str(i) for i in range(10)]


def _crop_and_center(rgba: np.ndarray) -> Image.Image | None:
    gray = rgba[:, :, :3].max(axis=2)
    rows = np.any(gray > 10, axis=1)
    cols = np.any(gray > 10, axis=0)
    if not rows.any():
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    h, w = rmax - rmin, cmax - cmin
    pad = max(int(max(h, w) * 0.2), 4)
    rmin = max(0, rmin - pad)
    rmax = min(gray.shape[0] - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(gray.shape[1] - 1, cmax + pad)

    cropped = gray[rmin : rmax + 1, cmin : cmax + 1]
    size = max(cropped.shape)
    square = np.zeros((size, size), dtype=np.uint8)
    y = (size - cropped.shape[0]) // 2
    x = (size - cropped.shape[1]) // 2
    square[y : y + cropped.shape[0], x : x + cropped.shape[1]] = cropped
    return Image.fromarray(square, mode="L")


class DigitCanvas:
    def __init__(
        self, detector, *, width: int = 280, height: int = 280, radius: int = 8
    ) -> None:
        self._detector = detector
        self._width = width
        self._height = height
        self._radius = radius
        self._drawing = [False]
        self._prev = [0, 0]
        self._widget = self._build()

    def show(self) -> None:
        display(self._widget)

    def _build(self) -> widgets.Widget:
        W, H, R = self._width, self._height, self._radius

        canvas = Canvas(width=W, height=H, sync_image_data=True)
        canvas.fill_style = "black"
        canvas.fill_rect(0, 0, W, H)

        bars = [
            widgets.FloatProgress(
                value=0,
                min=0,
                max=1,
                description=lbl,
                style={"bar_color": "#4a90d9"},
                layout=widgets.Layout(width="260px"),
            )
            for lbl in _LABELS
        ]

        def _update_bars(probs: list[float]) -> None:
            top = max(range(10), key=lambda i: probs[i])
            for i, (bar, p) in enumerate(zip(bars, probs)):
                bar.value = float(p)
                bar.style = {"bar_color": "#2ecc71" if i == top else "#4a90d9"}

        def _on_image_data(change):
            if self._drawing[0]:
                return
            raw = change["new"]
            if raw is None:
                return
            rgba = np.array(Image.open(io.BytesIO(raw)).convert("RGBA"))
            img = _crop_and_center(rgba)
            if img is None:
                return
            _update_bars(self._detector.probs(img))

        canvas.observe(_on_image_data, names=["image_data"])

        def _mouse_down(x, y):
            self._drawing[0] = True
            self._prev[:] = [x, y]
            with hold_canvas(canvas):
                canvas.fill_style = "white"
                canvas.fill_arc(x, y, R, 0, 2 * math.pi)

        def _mouse_move(x, y):
            if not self._drawing[0]:
                return
            with hold_canvas(canvas):
                canvas.stroke_style = "white"
                canvas.line_width = R * 2
                canvas.line_cap = "round"
                canvas.begin_path()
                canvas.move_to(self._prev[0], self._prev[1])
                canvas.line_to(x, y)
                canvas.stroke()
            self._prev[:] = [x, y]

        def _mouse_up(*_):
            self._drawing[0] = False
            canvas.fill_style = "black"
            canvas.fill_rect(0, 0, 1, 1)  # 1×1 trigger for image_data sync

        canvas.on_mouse_down(_mouse_down)
        canvas.on_mouse_move(_mouse_move)
        canvas.on_mouse_up(_mouse_up)

        btn_clear = widgets.Button(description="Clear")

        def _clear(_):
            canvas.clear()
            canvas.fill_style = "black"
            canvas.fill_rect(0, 0, W, H)
            for bar in bars:
                bar.value = 0
                bar.style = {"bar_color": "#4a90d9"}

        btn_clear.on_click(_clear)

        return widgets.HBox([widgets.VBox([canvas, btn_clear]), widgets.VBox(bars)])
