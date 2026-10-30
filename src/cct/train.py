"""Entrenamiento de YOLO (Ultralytics) con decisiones documentadas.

Entregable 1: resolucion de entrada, estrategia de aumento de datos y manejo de las
imagenes vacias como fondo. Todo se guarda en runs/<name>/decisions.json.

Presets de aumento de datos:
  baseline : solo flip horizontal (referencia para el ablation).
  strong   : mosaic + HSV + escala/traslacion + mixup suave + borrado aleatorio. Pensado para
             camaras trampa: el borrado aleatorio simula oclusion por vegetacion, el HSV cubre
             dia / noche infrarrojo / cambios de luz, y mosaic ayuda con animales pequenos.

Uso:  python -m cct.train --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

from .utils import add_common_args, ensure_dir, p, parse_with_config, set_seed

AUG_PRESETS = {
    "baseline": dict(hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0,
                     shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=0.0, mixup=0.0,
                     copy_paste=0.0, erasing=0.0, close_mosaic=0),
    "strong": dict(hsv_h=0.015, hsv_s=0.5, hsv_v=0.4, degrees=5.0, translate=0.1, scale=0.5,
                   shear=2.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.1,
                   copy_paste=0.0, erasing=0.4, close_mosaic=10),
}


def train(cfg: dict) -> Path:
    from ultralytics import YOLO

    t = cfg["train"]
    set_seed(cfg["seed"])
    data_yaml = p(cfg, "yolo") / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Falta {data_yaml}. Corre primero: python -m cct.build_dataset")
    runs = ensure_dir(p(cfg, "runs"))
    aug = AUG_PRESETS[t["aug_preset"]]

    kwargs = dict(
        data=str(data_yaml), imgsz=t["imgsz"], epochs=t["epochs"], batch=t["batch"],
        patience=t["patience"], workers=t["workers"], device=t["device"], cache=t["cache"],
        project=str(runs), name=t["name"], exist_ok=True, seed=cfg["seed"], deterministic=True,
        cos_lr=True, optimizer="auto", pretrained=True, plots=True, **aug,
    )
    decisions = {
        "modelo": t["model"], "resolucion_entrada": t["imgsz"], "aug_preset": t["aug_preset"],
        "aumento_de_datos": aug, "vacias_como_fondo": True,
        "train_empty_ratio": cfg["data"]["train_empty_ratio"],
        "rare_oversample_RFS": cfg["data"]["rare_oversample"], "rfs_threshold": cfg["data"]["rfs_threshold"],
        "hiperparametros_ultralytics": {k: v for k, v in kwargs.items() if k not in aug},
        "python": sys.version, "plataforma": platform.platform(),
    }
    try:
        import torch, ultralytics
        decisions["torch"] = torch.__version__
        decisions["ultralytics"] = ultralytics.__version__
    except Exception:
        pass
    out_dir = ensure_dir(runs / t["name"])
    (out_dir / "decisions.json").write_text(json.dumps(decisions, indent=1, default=str))
    try:  # congela las versiones para reproducir
        freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
        (out_dir / "pip_freeze.txt").write_text(freeze)
    except Exception:
        pass

    model = YOLO(t["model"])
    model.train(**kwargs)
    best = runs / t["name"] / "weights" / "best.pt"
    print("Pesos:", best)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args, cfg = parse_with_config(ap)
    train(cfg)


if __name__ == "__main__":
    main()
