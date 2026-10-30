"""Utilidades comunes: configuracion, semilla, rutas y lectura del manifiesto."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_config(path: str | os.PathLike, overrides: list[str] | None = None) -> dict:
    """Lee el YAML y aplica overrides tipo ``train.epochs=5`` (se parsean como YAML)."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for item in overrides or []:
        key, _, value = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--set", nargs="*", default=[], dest="overrides",
                        help="overrides, ej: --set train.epochs=5 train.device=cpu")
    return parser


def parse_with_config(parser: argparse.ArgumentParser):
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    return args, cfg


def root(cfg: dict) -> Path:
    return Path(cfg["paths"]["root"]).expanduser().resolve()


def p(cfg: dict, key: str) -> Path:
    """Ruta absoluta de paths.<key> (relativa a root)."""
    return root(cfg) / cfg["paths"][key]


def data_path(cfg: dict, key: str) -> Path:
    """Ruta absoluta de data.<key> (relativa a root salvo que ya sea absoluta)."""
    q = Path(cfg["data"][key])
    return q if q.is_absolute() else root(cfg) / q


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:  # torch es opcional para algunos scripts
        pass


def read_manifest(cfg: dict) -> pd.DataFrame:
    path = p(cfg, "yolo") / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Corre primero: python -m cct.build_dataset")
    df = pd.read_csv(path, dtype={"location": str, "seq_id": str})
    return df


def read_classes(cfg: dict) -> list[str]:
    path = p(cfg, "yolo") / "classes.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str | os.PathLike) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_yolo_labels(label_path: str | os.PathLike) -> np.ndarray:
    """Devuelve array (n, 5) con cls, cx, cy, w, h normalizados. Vacio si no hay etiqueta."""
    label_path = Path(label_path)
    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.zeros((0, 5), dtype=np.float64)
    arr = np.loadtxt(label_path, ndmin=2, dtype=np.float64)
    return arr.reshape(-1, 5)


def label_path_for(image_path: str | os.PathLike) -> Path:
    """Misma convencion que Ultralytics: .../images/... -> .../labels/....txt"""
    s = str(image_path)
    idx = s.rfind(os.sep + "images" + os.sep)
    if idx < 0:
        raise ValueError(f"La ruta no contiene /images/: {s}")
    return Path(s[:idx] + os.sep + "labels" + os.sep + s[idx + len(os.sep + "images" + os.sep):]).with_suffix(".txt")


def deep_copy(cfg: dict) -> dict:
    return copy.deepcopy(cfg)
