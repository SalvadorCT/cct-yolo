"""Evaluacion propia basada en pycocotools (mAP@0.5, mAP@0.5:0.95, por clase y por grupos).

Las cajas se evaluan en coordenadas normalizadas escaladas x1000. El IoU no cambia al
escalar cada eje por separado, asi que no necesitamos el tamano real de cada imagen.

Subconjuntos (tamano, oclusion): se pasa una mascara `ignore` sobre las cajas GT. Las GT
ignoradas no cuentan como positivos y las detecciones que caen sobre ellas no cuentan como
falsos positivos (misma idea que las areas small/medium/large de COCO).
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd

from .utils import label_path_for, read_yolo_labels


def load_gt(manifest_subset: pd.DataFrame) -> pd.DataFrame:
    """Cajas GT de un subconjunto: name, cls, cx, cy, w, h, xmin, ymin, xmax, ymax, rel_area, location."""
    rows = []
    for r in manifest_subset.itertuples():
        arr = read_yolo_labels(label_path_for(r.image_path))
        for c, cx, cy, w, h in arr:
            rows.append((r.name, int(c), cx, cy, w, h, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, w * h,
                         r.location, r.seq_id))
    cols = ["name", "cls", "cx", "cy", "w", "h", "xmin", "ymin", "xmax", "ymax", "rel_area", "location", "seq_id"]
    return pd.DataFrame(rows, columns=cols)


def coco_eval(gt: pd.DataFrame, pred: pd.DataFrame, image_names: list[str], n_classes: int,
              ignore: np.ndarray | None = None) -> dict:
    """Devuelve dict con map50, map5095 y DataFrame por clase (ap50, ap5095, recall50, n_gt)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    name2id = {n: i + 1 for i, n in enumerate(image_names)}
    ignore = np.zeros(len(gt), dtype=bool) if ignore is None else np.asarray(ignore, dtype=bool)
    S = 1000.0
    anns = []
    for k, r in enumerate(gt.itertuples()):
        if r.name not in name2id:
            continue
        w, h = r.w * S, r.h * S
        anns.append(dict(id=k + 1, image_id=name2id[r.name], category_id=int(r.cls),
                         bbox=[r.xmin * S, r.ymin * S, w, h], area=1e11 if ignore[k] else w * h, iscrowd=0))
    n_gt = np.zeros(n_classes, dtype=int)
    for k, r in enumerate(gt.itertuples()):
        if not ignore[k] and r.name in name2id:
            n_gt[int(r.cls)] += 1
    empty = pd.DataFrame({"class_id": range(n_classes), "ap50": np.nan, "ap5095": np.nan,
                          "recall50": np.nan, "n_gt": n_gt})
    pred = pred[pred["name"].isin(name2id)] if len(pred) else pred
    if len(anns) == 0 or len(pred) == 0 or n_gt.sum() == 0:
        e = empty.copy()
        e.loc[e.n_gt > 0, ["ap50", "ap5095", "recall50"]] = 0.0
        return dict(map50=0.0 if n_gt.sum() else np.nan, map5095=0.0 if n_gt.sum() else np.nan, per_class=e)

    ds = dict(images=[dict(id=i, width=int(S), height=int(S)) for i in name2id.values()],
              annotations=anns, categories=[dict(id=i, name=str(i)) for i in range(n_classes)])
    dets = [dict(image_id=name2id[r.name], category_id=int(r.cls), score=float(r.conf),
                 bbox=[r.xmin * S, r.ymin * S, (r.xmax - r.xmin) * S, (r.ymax - r.ymin) * S])
            for r in pred.itertuples()]
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = ds
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dets)
        E = COCOeval(coco_gt, coco_dt, "bbox")
        E.params.imgIds = list(name2id.values())
        E.params.areaRng = [[0, 1e10]]
        E.params.areaRngLbl = ["all"]
        E.params.maxDets = [100]
        E.evaluate()
        E.accumulate()
    prec = E.eval["precision"]   # [T, R, K, A, M]
    rec = E.eval["recall"]       # [T, K, A, M]
    ap50, ap5095, r50 = np.full(n_classes, np.nan), np.full(n_classes, np.nan), np.full(n_classes, np.nan)
    for k in range(n_classes):
        if n_gt[k] == 0:
            continue
        pk = prec[:, :, k, 0, 0]
        ap5095[k] = np.mean(np.where(pk > -1, pk, 0)) if (pk > -1).any() else 0.0
        p50 = pk[0]
        ap50[k] = np.mean(np.where(p50 > -1, p50, 0)) if (p50 > -1).any() else 0.0
        r50[k] = max(rec[0, k, 0, 0], 0.0)
    per_class = pd.DataFrame(dict(class_id=range(n_classes), ap50=ap50, ap5095=ap5095, recall50=r50, n_gt=n_gt))
    return dict(map50=float(np.nanmean(ap50)), map5095=float(np.nanmean(ap5095)), per_class=per_class)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ix0 = np.maximum(a[:, None, 0], b[None, :, 0]); iy0 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix1 = np.minimum(a[:, None, 2], b[None, :, 2]); iy1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter + 1e-12)


def detected_mask(gt: pd.DataFrame, pred: pd.DataFrame, conf: float, iou_thr: float = 0.5,
                  class_aware: bool = True) -> np.ndarray:
    """Para cada GT: True si alguna deteccion (conf >= conf, misma clase, IoU >= iou_thr) la cubre.
    Emparejamiento voraz uno a uno por confianza."""
    det = np.zeros(len(gt), dtype=bool)
    if len(gt) == 0 or len(pred) == 0:
        return det
    pr = pred[pred.conf >= conf]
    pgroups = {k: v for k, v in pr.groupby("name")}
    gt_idx = {k: np.array(v) for k, v in gt.reset_index(drop=True).groupby("name").groups.items()}
    for name, idxs in gt_idx.items():
        if name not in pgroups:
            continue
        pg = pgroups[name].sort_values("conf", ascending=False)
        gb = gt.iloc[idxs][["xmin", "ymin", "xmax", "ymax"]].to_numpy()
        pb = pg[["xmin", "ymin", "xmax", "ymax"]].to_numpy()
        iou = _iou_matrix(gb, pb)
        if class_aware:
            iou = np.where(gt.iloc[idxs]["cls"].to_numpy()[:, None] == pg["cls"].to_numpy()[None, :], iou, 0.0)
        used_p = np.zeros(len(pg), dtype=bool)
        order = np.argsort(-pg["conf"].to_numpy())
        for j in order:   # detecciones de mayor a menor confianza
            cand = np.where((iou[:, j] >= iou_thr) & (~det[idxs]))[0]
            if len(cand) == 0:
                continue
            best = cand[np.argmax(iou[cand, j])]
            det[idxs[best]] = True
    return det


def empty_image_stats(pred: pd.DataFrame, empty_names: list[str], conf: float) -> dict:
    """Falsos positivos sobre imagenes vacias: fraccion con >=1 deteccion y detecciones por imagen."""
    if len(empty_names) == 0:
        return dict(n_empty=0, frac_with_fp=np.nan, fp_per_image=np.nan)
    pr = pred[(pred.conf >= conf) & (pred["name"].isin(set(empty_names)))]
    counts = pr.groupby("name").size()
    return dict(n_empty=len(empty_names), frac_with_fp=len(counts) / len(empty_names),
                fp_per_image=float(counts.sum()) / len(empty_names))


def to_latex(df: pd.DataFrame, path, caption: str, label: str, float_fmt="%.3f") -> None:
    s = df.to_latex(index=False, float_format=float_fmt, na_rep="--", escape=True, column_format="l" + "r" * (df.shape[1] - 1))
    s = s.replace("\\begin{tabular}", "\\begin{tabular}", 1)
    body = ("% Tabla generada automaticamente por el codigo (no editar a mano)\n"
            "\\begin{table}[ht]\n\\centering\n\\small\n" + s + f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{table}}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
