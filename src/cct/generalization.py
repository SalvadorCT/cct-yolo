"""Entregable 3: generalizacion entre ubicaciones de camara (vistas vs nuevas).

1. Brecha de desempeno: mAP en test_seen (ubicaciones del entrenamiento) vs test_unseen (eccv_val).
2. Falsos positivos sobre imagenes vacias en cada caso.
3. AP por ubicacion nueva (dispersion entre camaras).
4. Evidencia de que el modelo mira el FONDO y no el animal:
     a) Grad-CAM: fraccion de la energia del mapa que cae dentro de la caja del animal
        (se compara con el area de la caja: si el mapa fuera uniforme, energia ~ area).
     b) Prueba de enmascarado: se tapa la caja del animal con gris y se mide en cuantas imagenes
        el modelo SIGUE detectando esa especie. Si sigue detectando, esta usando el fondo.

Salidas: reports/tables/generalization_*.csv, reports/figures/gradcam_*.png, generalization_gap.png

Uso:  python -m cct.generalization --config configs/config.yaml
"""
from __future__ import annotations

import argparse

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .eval_utils import coco_eval, detected_mask, empty_image_stats, load_gt
from .evaluate import get_preds, load_model
from .gradcam import YoloGradCAM, energy_in_box, overlay
from .utils import add_common_args, ensure_dir, p, parse_with_config, read_classes, read_manifest, set_seed


def mask_boxes(bgr, boxes_xyxyn, fill=114):
    out = bgr.copy()
    H, W = out.shape[:2]
    for x0, y0, x1, y1 in boxes_xyxyn:
        out[int(y0 * H):int(y1 * H) + 1, int(x0 * W):int(x1 * W) + 1] = fill
    return out


def run(cfg: dict, weights: str | None = None) -> None:
    set_seed(cfg["seed"])
    classes, manifest = read_classes(cfg), read_manifest(cfg)
    model, _ = load_model(cfg, weights)
    tag, e = cfg["train"]["name"], cfg["eval"]
    rep = p(cfg, "reports")
    tables, figs = ensure_dir(rep / "tables"), ensure_dir(rep / "figures")

    # ---------- 1-3: brecha y falsos positivos ----------
    data = {}
    gaps = []
    for subset in ("test_seen", "test_unseen"):
        sub, pred = get_preds(cfg, model, manifest, subset, classes, tag)
        gt = load_gt(sub)
        res = coco_eval(gt, pred, sub["name"].tolist(), len(classes))
        det = detected_mask(gt, pred, e["conf_operating"])
        fp = empty_image_stats(pred, sub.loc[sub.is_empty == 1, "name"].tolist(), e["conf_operating"])
        gaps.append(dict(subset=subset, map50=res["map50"], map5095=res["map5095"],
                         recall_at_conf=float(det.mean()) if len(det) else np.nan, **fp))
        data[subset] = (sub, pred, gt, res)
    gap = pd.DataFrame(gaps)
    gap.to_csv(tables / "generalization_gap.csv", index=False)
    s, u = gap.iloc[0], gap.iloc[1]
    drop = pd.DataFrame([dict(metrica=m, vistas=s[m], nuevas=u[m], brecha_abs=s[m] - u[m],
                              caida_relativa_pct=100 * (s[m] - u[m]) / s[m] if s[m] else np.nan)
                         for m in ("map50", "map5095", "recall_at_conf", "frac_with_fp")])
    drop.to_csv(tables / "generalization_drop.csv", index=False)
    print(drop.to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    labs = ["mAP@0.5", "mAP@0.5:0.95", f"Recall@conf{e['conf_operating']}"]
    xs = np.arange(3)
    ax.bar(xs - 0.18, [s.map50, s.map5095, s.recall_at_conf], 0.36, label="ubicaciones vistas")
    ax.bar(xs + 0.18, [u.map50, u.map5095, u.recall_at_conf], 0.36, label="ubicaciones nuevas")
    ax.set_xticks(xs, labs, fontsize=8); ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Brecha de generalizacion entre ubicaciones", fontsize=10)
    fig.tight_layout(); fig.savefig(figs / "generalization_gap.png", dpi=200); plt.close(fig)

    # AP por ubicacion nueva
    sub_u, pred_u, gt_u, _ = data["test_unseen"]
    rows = []
    for loc, g in sub_u.groupby("location"):
        names = g["name"].tolist()
        gt_l = gt_u[gt_u["name"].isin(names)]
        r = coco_eval(gt_l, pred_u[pred_u["name"].isin(names)], names, len(classes))
        rows.append(dict(location=loc, n_images=len(g), n_boxes=len(gt_l), map50=r["map50"]))
    pd.DataFrame(rows).sort_values("map50").to_csv(tables / "generalization_per_location_unseen.csv", index=False)

    # ---------- 4: Grad-CAM + enmascarado ----------
    g = cfg["gradcam"]
    cam_engine = YoloGradCAM(model, g["layer_index"], cfg["train"]["imgsz"], cfg["train"]["device"])
    rng = np.random.default_rng(cfg["seed"])
    records, panels = [], {}
    for subset in ("test_seen", "test_unseen"):
        sub, pred, gt, _ = data[subset]
        cand = sub[sub.is_empty == 0]["name"].tolist()
        rng.shuffle(cand)
        taken = 0
        panels[subset] = []
        for name in cand:
            if taken >= g["n_images"]:
                break
            row = sub[sub.name == name].iloc[0]
            gts = gt[gt.name == name]
            if len(gts) == 0:
                continue
            bgr = cv2.imread(row.image_path)
            if bgr is None:
                continue
            # clase objetivo = la del animal mas grande
            big = gts.sort_values("rel_area", ascending=False).iloc[0]
            cam, cls_used, sc = cam_engine(bgr, int(big.cls))
            en = energy_in_box(cam, (big.xmin, big.ymin, big.xmax, big.ymax))
            # prueba de enmascarado: se tapan TODAS las cajas del animal
            masked = mask_boxes(bgr, gts[["xmin", "ymin", "xmax", "ymax"]].to_numpy())
            res = model.predict(masked, imgsz=cfg["train"]["imgsz"], conf=e["conf_operating"], verbose=False,
                                device=cfg["train"]["device"])[0]
            persists = bool(len(res.boxes) and (res.boxes.cls.cpu().numpy().astype(int) == int(big.cls)).any())
            res0 = model.predict(bgr, imgsz=cfg["train"]["imgsz"], conf=e["conf_operating"], verbose=False,
                                 device=cfg["train"]["device"])[0]
            detected_orig = bool(len(res0.boxes) and (res0.boxes.cls.cpu().numpy().astype(int) == int(big.cls)).any())
            records.append(dict(subset=subset, name=name, location=row.location, cls=classes[int(big.cls)],
                                box_area=float(big.rel_area), energy_in_box=en, cam_score=sc,
                                detected_original=detected_orig, detected_after_masking=persists))
            if len(panels[subset]) < 4:
                panels[subset].append((overlay(bgr, cam), bgr, big, row.location, classes[int(big.cls)], en))
            taken += 1
    cam_engine.close()
    rec = pd.DataFrame(records)
    rec.to_csv(tables / "generalization_gradcam_records.csv", index=False)
    if len(rec):
        agg = rec.groupby("subset").agg(n=("name", "size"), energia_en_caja=("energy_in_box", "mean"),
                                         area_caja=("box_area", "mean"),
                                         detecta_original=("detected_original", "mean"),
                                         detecta_tras_enmascarar=("detected_after_masking", "mean")).reset_index()
        agg["energia_sobre_area"] = agg["energia_en_caja"] / agg["area_caja"]
        agg.to_csv(tables / "generalization_gradcam_summary.csv", index=False)
        print(agg.to_string(index=False))

    # figura: 4 ejemplos vistas / 4 nuevas
    n = max(len(panels.get("test_seen", [])), len(panels.get("test_unseen", [])))
    if n:
        fig, axes = plt.subplots(2, n, figsize=(3.0 * n, 5.6), squeeze=False)
        for r, (subset, title) in enumerate((("test_seen", "ubicacion vista"), ("test_unseen", "ubicacion nueva"))):
            for c in range(n):
                ax = axes[r][c]; ax.axis("off")
                if c < len(panels[subset]):
                    ov, bgr, big, loc, cname, en = panels[subset][c]
                    H, W = bgr.shape[:2]
                    ax.imshow(cv2.cvtColor(ov, cv2.COLOR_BGR2RGB))
                    ax.add_patch(plt.Rectangle((big.xmin * W, big.ymin * H), big.w * W, big.h * H, fill=False, ec="white", lw=1.5))
                    ax.set_title(f"{title}\n{cname} | en caja: {en:.2f}", fontsize=7)
        fig.tight_layout(); fig.savefig(figs / "gradcam_seen_vs_unseen.png", dpi=170); plt.close(fig)
    print("Listo: reports/tables/generalization_*.csv y reports/figures/gradcam_seen_vs_unseen.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--weights", default=None)
    args, cfg = parse_with_config(ap)
    run(cfg, args.weights)


if __name__ == "__main__":
    main()
