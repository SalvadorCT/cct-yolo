"""Entregable 4: robustez por tamano relativo de la caja y por oclusion estimada.

Cada GT se asigna a un grupo (terciles). Para evaluar un grupo se ignoran las GT de los demas
(ver eval_utils.coco_eval), asi las detecciones de otros animales no cuentan como falsos positivos.
Se reporta mAP@0.5, mAP@0.5:0.95 y recall a la confianza operativa, y la caida entre grupos.

Salidas: reports/tables/robustness_{size,occlusion}_<subset>.csv, robustness_drop.csv,
         reports/figures/robustness_*.png
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import occlusion
from .eval_utils import coco_eval, detected_mask, load_gt
from .evaluate import get_preds, load_model
from .utils import add_common_args, ensure_dir, p, parse_with_config, read_classes, read_manifest, set_seed

SIZE_LABELS = ["pequena", "mediana", "grande"]
OCC_LABELS = ["baja", "media", "alta"]


def eval_groups(gt, pred, names, nc, conf, col, labels):
    rows = []
    det = detected_mask(gt.reset_index(drop=True), pred, conf)
    for lab in labels:
        sel = (gt[col] == lab).to_numpy()
        if sel.sum() == 0:
            continue
        r = coco_eval(gt, pred, names, nc, ignore=~sel)
        rows.append(dict(grupo=lab, n_cajas=int(sel.sum()), map50=r["map50"], map5095=r["map5095"],
                         recall_conf=float(det[sel].mean())))
    return pd.DataFrame(rows)


def drop_row(df, factor, subset, best, worst):
    a, b = df[df.grupo == best], df[df.grupo == worst]
    if a.empty or b.empty:
        return []
    out = []
    for m in ("map50", "map5095", "recall_conf"):
        va, vb = float(a[m].iloc[0]), float(b[m].iloc[0])
        out.append(dict(factor=factor, subset=subset, metrica=m, mejor_grupo=best, peor_grupo=worst,
                        mejor=va, peor=vb, caida_abs=va - vb, caida_rel_pct=100 * (va - vb) / va if va else np.nan))
    return out


def run(cfg: dict, weights: str | None = None) -> None:
    set_seed(cfg["seed"])
    classes, manifest = read_classes(cfg), read_manifest(cfg)
    model, _ = load_model(cfg, weights)
    tag, e = cfg["train"]["name"], cfg["eval"]
    rep = p(cfg, "reports")
    tables, figs = ensure_dir(rep / "tables"), ensure_dir(rep / "figures")
    drops, allres = [], {}

    # umbrales de tamano: terciles sobre ambos conjuntos de prueba juntos (mismos cortes para todos)
    both = load_gt(manifest[manifest.subset.isin(["test_seen", "test_unseen"])])
    q = both["rel_area"].quantile([1 / 3, 2 / 3]).to_numpy()
    pd.DataFrame([dict(corte_pequena_mediana=q[0], corte_mediana_grande=q[1])]).to_csv(tables / "robustness_size_cuts.csv", index=False)

    for subset in ("test_seen", "test_unseen"):
        sub, pred = get_preds(cfg, model, manifest, subset, classes, tag)
        gt = load_gt(sub).reset_index(drop=True)
        names = sub["name"].tolist()
        gt["size_group"] = np.where(gt.rel_area <= q[0], "pequena", np.where(gt.rel_area <= q[1], "mediana", "grande"))
        occ = occlusion.compute(cfg, gt, manifest, cache=str(rep / "preds" / f"occ_{subset}.csv"))
        gt["occ_group"] = occ["occ_group"].to_numpy()
        gt["occ"] = occ["occ"].to_numpy()
        gt["trunc"] = occ["trunc"].to_numpy()

        s = eval_groups(gt, pred, names, len(classes), e["conf_operating"], "size_group", SIZE_LABELS)
        o = eval_groups(gt, pred, names, len(classes), e["conf_operating"], "occ_group", OCC_LABELS)
        t = eval_groups(gt.assign(tr=np.where(gt.trunc == 1, "cortado", "completo")), pred, names, len(classes),
                        e["conf_operating"], "tr", ["completo", "cortado"])
        s.to_csv(tables / f"robustness_size_{subset}.csv", index=False)
        o.to_csv(tables / f"robustness_occlusion_{subset}.csv", index=False)
        t.to_csv(tables / f"robustness_truncation_{subset}.csv", index=False)
        drops += drop_row(s, "tamano", subset, "grande", "pequena")
        drops += drop_row(o, "oclusion", subset, "baja", "alta")
        drops += drop_row(t, "truncamiento", subset, "completo", "cortado")
        allres[subset] = (s, o)
        # cruce tamano x oclusion (recall)
        det = detected_mask(gt, pred, e["conf_operating"])
        piv = gt.assign(det=det).pivot_table(index="size_group", columns="occ_group", values="det", aggfunc="mean")
        piv = piv.reindex(index=SIZE_LABELS, columns=OCC_LABELS)
        piv.to_csv(tables / f"robustness_cross_recall_{subset}.csv")
        print(subset, "\n", s.to_string(index=False), "\n", o.to_string(index=False))

    d = pd.DataFrame(drops)
    d.to_csv(tables / "robustness_drop.csv", index=False)
    print(d.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, (idx, labels, title) in zip(axes, ((0, SIZE_LABELS, "tamano relativo de la caja"), (1, OCC_LABELS, "oclusion estimada"))):
        for k, subset in enumerate(("test_seen", "test_unseen")):
            df = allres[subset][idx].set_index("grupo").reindex(labels)
            ax.bar(np.arange(3) + (k - 0.5) * 0.36, df["map50"].fillna(0), 0.36,
                   label="vistas" if subset == "test_seen" else "nuevas")
        ax.set_xticks(range(3), labels); ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("mAP@0.5"); axes[0].legend(fontsize=8); axes[0].set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(figs / "robustness_groups.png", dpi=200); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--weights", default=None)
    args, cfg = parse_with_config(ap)
    run(cfg, args.weights)


if __name__ == "__main__":
    main()
