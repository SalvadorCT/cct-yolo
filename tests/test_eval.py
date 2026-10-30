"""Pruebas unitarias de las metricas con predicciones fabricadas (no necesitan modelo)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cct.eval_utils import coco_eval, detected_mask, empty_image_stats  # noqa: E402
from cct.monitoring import apply_rules, build_report  # noqa: E402


def make_gt():
    rows = []
    for i in range(20):
        c = i % 2
        x0, y0 = 0.1 + 0.02 * (i % 5), 0.2
        w, h = (0.1, 0.1) if i < 10 else (0.4, 0.4)
        rows.append((f"im{i}", c, x0 + w / 2, y0 + h / 2, w, h, x0, y0, x0 + w, y0 + h, w * h, "1", f"s{i // 2}"))
    cols = ["name", "cls", "cx", "cy", "w", "h", "xmin", "ymin", "xmax", "ymax", "rel_area", "location", "seq_id"]
    return pd.DataFrame(rows, columns=cols)


def perfect_pred(gt, conf=0.9):
    return gt[["name", "cls", "xmin", "ymin", "xmax", "ymax"]].assign(conf=conf)


def test_perfect():
    gt = make_gt(); names = gt.name.tolist() + ["empty0"]
    r = coco_eval(gt, perfect_pred(gt), names, 2)
    assert r["map50"] > 0.99 and r["map5095"] > 0.99, r["map50"]
    assert detected_mask(gt, perfect_pred(gt), 0.25).all()


def test_wrong_class_and_empty():
    gt = make_gt(); pr = perfect_pred(gt); pr["cls"] = 1 - pr["cls"]
    assert coco_eval(gt, pr, gt.name.tolist(), 2)["map50"] < 0.01
    assert not detected_mask(gt, pr, 0.25).any()
    fp = empty_image_stats(perfect_pred(gt).assign(name="e0").head(3), ["e0", "e1"], 0.25)
    assert fp["frac_with_fp"] == 0.5 and fp["fp_per_image"] == 1.5


def test_ignore_groups():
    """Si el modelo acierta solo las cajas grandes: AP del grupo grande = 1, del pequeno ~ 0, y sin FPs cruzados."""
    gt = make_gt(); big = (gt.rel_area > 0.05).to_numpy()
    pr = perfect_pred(gt)[big]
    rb = coco_eval(gt, pr, gt.name.tolist(), 2, ignore=~big)
    rs = coco_eval(gt, pr, gt.name.tolist(), 2, ignore=big)
    assert rb["map50"] > 0.99, rb["map50"]
    assert rs["map50"] < 0.01, rs["map50"]


def test_monitoring_rules():
    pred = pd.DataFrame(dict(name=["a", "a", "b", "c"], cls=[0, 0, 1, 1], conf=[0.9, 0.8, 0.2, 0.5],
                             xmin=0, ymin=0, xmax=1, ymax=1))
    classes = ["deer", "fox"]
    kept = apply_rules(pred, classes, dict(conf_threshold=0.3, priority_classes=["fox"], priority_conf_threshold=0.15))
    assert set(kept.name) == {"a", "b", "c"}          # fox en 'b' (0.2) pasa por ser prioritaria
    photos = pd.DataFrame(dict(name=list("abcd"), datetime=["2013-01-01 10:00:00"] * 3 + ["2013-01-02 10:00:00"],
                               seq_id=["s1", "s1", "s2", "s3"]))
    act, summ = build_report(photos, kept, "day")
    row = act[(act.periodo == "2013-01-01") & (act.species == "deer")].iloc[0]
    assert row.detecciones == 2 and row.individuos == 2 and row.secuencias == 1   # 2 cajas en la misma foto = 2 individuos
    s2 = summ[summ.periodo == "2013-01-02"].iloc[0]
    assert s2.fotos_vacias == 1 and s2.pct_fauna == 0


if __name__ == "__main__":
    for k, v in list(globals().items()):
        if k.startswith("test_"):
            v(); print("ok", k)
