"""Grad-CAM propio para detectores YOLO (Ultralytics), sin librerias externas.

Idea: tomamos las activaciones de una capa del backbone (A) y el gradiente de un puntaje
escalar de la deteccion respecto a ellas (dS/dA). Los pesos de cada canal son el promedio
espacial del gradiente; el mapa es ReLU(sum_k alpha_k * A_k). Esto es Grad-CAM (Selvaraju et al., 2017)
aplicado al "puntaje de clase mas alto entre todas las celdas", que es lo que un detector devuelve.

Si el puntaje de la clase objetivo no depende de la capa elegida, el mapa sale en cero; en ese caso
cambia `gradcam.layer_index` (para yolov8 la capa 9 es el SPPF final del backbone).
"""
from __future__ import annotations

import copy

import cv2
import numpy as np
import torch


def letterbox(img: np.ndarray, size: int, color=(114, 114, 114)):
    h, w = img.shape[:2]
    s = min(size / h, size / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, s, left, top, nh, nw


class YoloGradCAM:
    def __init__(self, yolo, layer_index: int = 9, imgsz: int = 640, device: str | int = "cpu"):
        self.net = copy.deepcopy(yolo.model).float()   # copia propia: los hooks no tocan el modelo original
        self.device = torch.device("cuda:0" if (str(device) not in ("cpu", "-1") and torch.cuda.is_available()) else "cpu")
        self.net.to(self.device).eval()
        for prm in self.net.parameters():
            prm.requires_grad_(True)
        self.imgsz = imgsz
        layers = self.net.model
        if layer_index >= len(layers) or layer_index < -len(layers):
            raise IndexError(f"layer_index {layer_index} fuera de rango (la red tiene {len(layers)} capas)")
        self.layer = layers[layer_index]
        self.act = None
        self._h = self.layer.register_forward_hook(self._save)

    def _save(self, module, inp, out):
        self.act = out if isinstance(out, torch.Tensor) else out[0]
        self.act.retain_grad()

    def close(self):
        self._h.remove()

    @staticmethod
    def _scores(out) -> torch.Tensor:
        y = out[0] if isinstance(out, (tuple, list)) else out
        if isinstance(y, dict):
            y = y.get("one2many", next(iter(y.values())))
        return y[0, 4:, :]   # (nc, n_anclas): puntajes de clase ya con sigmoide

    @torch.enable_grad()
    def __call__(self, bgr: np.ndarray, target_cls: int | None = None):
        """Devuelve (cam [H,W] en 0-1 sobre la imagen original, clase usada, puntaje maximo)."""
        H, W = bgr.shape[:2]
        canvas, s, left, top, nh, nw = letterbox(bgr, self.imgsz)
        x = torch.from_numpy(canvas[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
        x.requires_grad_(True)
        self.net.zero_grad(set_to_none=True)
        out = self.net(x)
        sc = self._scores(out)
        if target_cls is None:
            target_cls = int(sc.max(dim=1).values.argmax())
        score = sc[target_cls].max()
        score.backward()
        A, G = self.act[0], self.act.grad[0] if self.act.grad is not None else None
        if G is None:
            return np.zeros((H, W), np.float32), target_cls, float(score)
        alpha = G.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((alpha * A).sum(0)).detach().cpu().numpy()
        self.act = None
        cam = cv2.resize(cam, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        cam = cam[top:top + nh, left:left + nw]
        cam = cv2.resize(cam, (W, H), interpolation=cv2.INTER_LINEAR)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32), target_cls, float(score.detach())


def overlay(bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 1 - alpha, heat, alpha, 0)


def energy_in_box(cam: np.ndarray, box_xyxyn) -> float:
    """Fraccion de la energia del mapa que cae dentro de la caja (coordenadas normalizadas)."""
    H, W = cam.shape
    x0, y0, x1, y1 = box_xyxyn
    sl = cam[int(max(0, y0) * H):int(min(1, y1) * H) + 1, int(max(0, x0) * W):int(min(1, x1) * W) + 1]
    tot = cam.sum()
    return float(sl.sum() / tot) if tot > 0 else float("nan")
