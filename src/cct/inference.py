"""Inferencia en lote con cache en disco (se reutiliza en evaluacion, robustez y monitoreo)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PRED_COLS = ["name", "cls", "conf", "xmin", "ymin", "xmax", "ymax"]


def predict_images(model, image_paths: list[str], names: list[str], imgsz: int, conf: float,
                   iou: float, batch: int, device) -> pd.DataFrame:
    """Devuelve un DataFrame con una fila por deteccion (cajas normalizadas xyxy 0-1)."""
    rows = []
    for i in tqdm(range(0, len(image_paths), batch), desc="inferencia"):
        chunk = image_paths[i:i + batch]
        res = model.predict(chunk, imgsz=imgsz, conf=conf, iou=iou, device=device,
                            verbose=False, max_det=100, stream=False)
        for pth, r in zip(chunk, res):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxyn = r.boxes.xyxyn.cpu().numpy()
            cl = r.boxes.cls.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            for b, c, s in zip(xyxyn, cl, cf):
                rows.append((Path(pth).name, int(c), float(s), *map(float, b)))
    return pd.DataFrame(rows, columns=PRED_COLS)


def cached_predictions(model, manifest_subset: pd.DataFrame, names: list[str], cache_file: Path,
                       imgsz: int, conf: float, iou: float, batch: int, device, force=False) -> pd.DataFrame:
    if cache_file.exists() and not force:
        return pd.read_csv(cache_file)
    df = predict_images(model, manifest_subset["image_path"].tolist(), names, imgsz, conf, iou, batch, device)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_file, index=False)
    return df
