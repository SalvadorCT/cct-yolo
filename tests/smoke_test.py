"""Prueba de humo en CPU con datos sinteticos: build -> train (1 epoca) -> evaluate -> generalization ->
robustness -> monitoring.   Uso:  python tests/smoke_test.py [directorio_temporal]"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from cct import build_dataset, evaluate, generalization, monitoring, robustness, train  # noqa: E402
from cct.utils import load_config  # noqa: E402
from make_synthetic import make  # noqa: E402


def main():
    tmp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="cct_smoke_"))
    tmp.mkdir(parents=True, exist_ok=True)
    make(tmp)
    ov = [f"paths.root={tmp}", "train.model=yolov8n.yaml", "train.imgsz=160", "train.epochs=1", "train.batch=8",
          "train.workers=0", "train.device=cpu", "train.name=smoke", "train.patience=5",
          "data.images_dir=data/raw/images", "gradcam.n_images=3", "robustness.bg_samples_per_location=5",
          "eval.batch=8"]
    cfg = load_config(str(REPO / "configs" / "config.yaml"), ov)
    build_dataset.build(cfg)
    train.train(cfg)
    w = tmp / "runs" / "smoke" / "weights" / "best.pt"
    assert w.exists(), f"no hay pesos en {w}"
    evaluate.evaluate(cfg, str(w))
    generalization.run(cfg, str(w))
    robustness.run(cfg, str(w))
    monitoring.run(cfg, str(w))
    print("\nSMOKE TEST OK ->", tmp)


if __name__ == "__main__":
    main()
