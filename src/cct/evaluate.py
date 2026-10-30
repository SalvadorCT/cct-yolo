"""Entregable 2: mAP@0.5 y mAP@0.5:0.95 por especie, y efecto del desbalance.

Salidas (reports/):
  tables/summary.csv                      mAP global por subconjunto
  tables/per_class_<subset>.csv / .tex    AP por especie + instancias de entrenamiento
  tables/common_vs_rare_<subset>.csv      promedio de AP en especies comunes vs raras
  figures/ap_vs_train_instances_<subset>.png

Uso:  python -m cct.evaluate --config configs/config.yaml [--weights ruta.pt]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .eval_utils import coco_eval, empty_image_stats, load_gt, to_latex
from .inference import cached_predictions
from .utils import (add_common_args, ensure_dir, p, parse_with_config, read_classes, read_manifest, set_seed)


def load_model(cfg: dict, weights: str | None):
    from ultralytics import YOLO
    w = Path(weights) if weights else p(cfg, "runs") / cfg["train"]["name"] / "weights" / "best.pt"
    if not w.exists():
        raise FileNotFoundError(f"No encuentro los pesos: {w}")
    return YOLO(str(w)), w


def get_preds(cfg, model, manifest, subset, classes, tag):
    e = cfg["eval"]
    sub = manifest[manifest.subset == subset]
    cache = p(cfg, "reports") / "preds" / f"{tag}_{subset}.csv"
    return sub, cached_predictions(model, sub, classes, cache, cfg["train"]["imgsz"], e["conf_low"],
                                   e["iou_nms"], e["batch"], cfg["train"]["device"])


def evaluate(cfg: dict, weights: str | None = None, subsets=("test_seen", "test_unseen")) -> pd.DataFrame:
    set_seed(cfg["seed"])
    classes = read_classes(cfg)
    manifest = read_manifest(cfg)
    model, wpath = load_model(cfg, weights)
    tag = cfg["train"]["name"]
    rep = p(cfg, "reports")
    tables, figs = ensure_dir(rep / "tables"), ensure_dir(rep / "figures")
    counts = pd.read_csv(p(cfg, "yolo") / "class_counts.csv", index_col=0)
    n_train = counts["train"].reindex(classes).fillna(0).astype(int).to_numpy()
    summary = []

    for subset in subsets:
        sub, pred = get_preds(cfg, model, manifest, subset, classes, tag)
        gt = load_gt(sub)
        res = coco_eval(gt, pred, sub["name"].tolist(), len(classes))
        pc = res["per_class"].copy()
        pc.insert(1, "especie", classes)
        pc["n_train"] = n_train
        pc = pc.drop(columns="class_id")
        pc.to_csv(tables / f"per_class_{subset}.csv", index=False)
        show = pc.rename(columns={"especie": "Especie", "ap50": "AP50", "ap5095": "AP50-95",
                                  "recall50": "Recall@0.5", "n_gt": "Inst. test", "n_train": "Inst. train"})
        to_latex(show.sort_values("Inst. train", ascending=False), tables / f"per_class_{subset}.tex",
                 caption=f"AP por especie en {subset.replace('_', ' ')}.", label=f"tab:perclass_{subset}")

        # --- efecto del desbalance: comunes vs raras ---
        valid = pc.dropna(subset=["ap50"])
        thr = valid["n_train"].quantile(cfg["eval"]["rare_quantile"]) if len(valid) else 0
        valid = valid.assign(grupo=np.where(valid["n_train"] <= thr, "rara", "comun"))
        cr = valid.groupby("grupo")[["ap50", "ap5095", "n_train"]].mean().reset_index()
        cr.to_csv(tables / f"common_vs_rare_{subset}.csv", index=False)
        rho = spearmanr(np.log1p(valid["n_train"]), valid["ap50"]) if len(valid) > 2 else None

        # --- grafico AP vs numero de instancias de entrenamiento ---
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.scatter(valid["n_train"] + 1, valid["ap50"], s=40, c=np.where(valid["grupo"] == "rara", "tab:red", "tab:blue"))
        for r in valid.itertuples():
            ax.annotate(r.especie, (r.n_train + 1, r.ap50), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("instancias de entrenamiento (+1, escala log)")
        ax.set_ylabel("AP@0.5")
        title = f"{subset}: AP vs frecuencia de la especie"
        if rho is not None:
            title += f"  (Spearman={rho.statistic:.2f})"
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figs / f"ap_vs_train_instances_{subset}.png", dpi=200)
        plt.close(fig)

        empties = sub.loc[sub.is_empty == 1, "name"].tolist()
        fp = empty_image_stats(pred, empties, cfg["eval"]["conf_operating"])
        summary.append(dict(subset=subset, n_images=len(sub), n_boxes=len(gt), map50=res["map50"],
                            map5095=res["map5095"], spearman_logn_ap50=getattr(rho, "statistic", np.nan), **fp))
        print(f"[{subset}] mAP50={res['map50']:.3f}  mAP50-95={res['map5095']:.3f}")
    sm = pd.DataFrame(summary)
    sm.to_csv(tables / "summary.csv", index=False)
    print(sm.to_string(index=False))
    return sm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--weights", default=None)
    args, cfg = parse_with_config(ap)
    evaluate(cfg, args.weights)


if __name__ == "__main__":
    main()
