"""Entregable 5: modulo de monitoreo ecologico.

Entrada : secuencia de fotos de UNA camara (por defecto, una ubicacion de test_unseen).
Reglas  : umbral de confianza por clase (menor para especies prioritarias, para no perder falsos
          negativos costosos); consolidacion por secuencia (individuos = maximo de cajas de esa
          especie en un mismo disparo de la secuencia, para no contar el mismo animal varias veces).
Salida  : reports/monitoring/<camara>_activity_<periodo>.csv  (tabla especie x periodo)
          reports/monitoring/<camara>_summary_<periodo>.csv   (resumen por periodo)
          reports/monitoring/<camara>_report.md y una figura de actividad.

Uso: python -m cct.monitoring --config configs/config.yaml [--location 33] [--period day|week|hour]
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .evaluate import get_preds, load_model
from .utils import add_common_args, ensure_dir, p, parse_with_config, read_classes, read_manifest, set_seed

FREQ = {"hour": "%Y-%m-%d %H:00", "day": "%Y-%m-%d", "week": "%G-W%V", "month": "%Y-%m"}


def apply_rules(pred: pd.DataFrame, classes: list[str], mcfg: dict) -> pd.DataFrame:
    thr = np.full(len(classes), mcfg["conf_threshold"], dtype=float)
    for c in mcfg.get("priority_classes", []) or []:
        if c in classes:
            thr[classes.index(c)] = mcfg["priority_conf_threshold"]
    keep = pred[pred.conf.to_numpy() >= thr[pred.cls.to_numpy().astype(int)]].copy() if len(pred) else pred.copy()
    keep["species"] = [classes[int(c)] for c in keep.cls]
    return keep


def build_report(photos: pd.DataFrame, det: pd.DataFrame, period: str):
    """photos: name, datetime, seq_id. det: detecciones ya filtradas (name, species)."""
    ph = photos.copy()
    ph["dt"] = pd.to_datetime(ph["datetime"], errors="coerce")
    ph["periodo"] = ph["dt"].dt.strftime(FREQ[period]).fillna("sin_fecha")
    cnt = det.groupby(["name", "species"]).size().rename("n_cajas").reset_index()
    cnt = cnt.merge(ph[["name", "seq_id", "periodo"]], on="name", how="left")
    # individuos: maximo de cajas por especie dentro de la secuencia
    ind = cnt.groupby(["seq_id", "periodo", "species"])["n_cajas"].max().rename("individuos").reset_index()
    seqs = cnt.groupby(["periodo", "species"])["seq_id"].nunique().rename("secuencias")
    act = (cnt.groupby(["periodo", "species"]).agg(detecciones=("n_cajas", "sum"), fotos=("name", "nunique"))
           .join(ind.groupby(["periodo", "species"])["individuos"].sum()).join(seqs).reset_index())
    ph["con_fauna"] = ph["name"].isin(set(det["name"]))
    summ = ph.groupby("periodo").agg(fotos=("name", "size"), fotos_con_fauna=("con_fauna", "sum")).reset_index()
    summ["fotos_vacias"] = summ.fotos - summ.fotos_con_fauna
    summ["pct_fauna"] = 100 * summ.fotos_con_fauna / summ.fotos
    summ["pct_vacias"] = 100 - summ["pct_fauna"]
    summ["especies_distintas"] = summ["periodo"].map(det.merge(ph[["name", "periodo"]], on="name").groupby("periodo")["species"].nunique()).fillna(0).astype(int)
    summ["detecciones"] = summ["periodo"].map(act.groupby("periodo")["detecciones"].sum()).fillna(0).astype(int)
    return act, summ


def run(cfg: dict, weights=None, location=None, subset="test_unseen", period=None) -> None:
    set_seed(cfg["seed"])
    classes, manifest = read_classes(cfg), read_manifest(cfg)
    m = cfg["monitoring"]
    period = period or m["period"]
    model, _ = load_model(cfg, weights)
    sub, pred = get_preds(cfg, model, manifest, subset, classes, cfg["train"]["name"])
    if location is None:
        location = sub["location"].value_counts().index[0]
    photos = sub[sub.location == str(location)].sort_values("datetime")
    if photos.empty:
        raise SystemExit(f"Sin fotos para la ubicacion {location} en {subset}")
    det = apply_rules(pred[pred.name.isin(photos.name)], classes, m)
    act, summ = build_report(photos, det, period)
    out = ensure_dir(p(cfg, "reports") / "monitoring")
    act.to_csv(out / f"loc{location}_activity_{period}.csv", index=False)
    summ.to_csv(out / f"loc{location}_summary_{period}.csv", index=False)

    # comparacion opcional con la verdad (solo para evaluar el modulo)
    gt_line = ""
    gt_species = set()
    from .eval_utils import load_gt
    g = load_gt(photos[photos.is_empty == 0])
    if len(g):
        gt_species = {classes[int(c)] for c in g.cls}
        pred_species = set(det.species)
        gt_line = (f"Especies en la verdad: {len(gt_species)} | detectadas: {len(pred_species)} | "
                   f"acierto de la lista: {len(gt_species & pred_species)}/{len(gt_species)} "
                   f"(falsas especies: {sorted(pred_species - gt_species)}).")
    total = summ[["fotos", "fotos_con_fauna", "fotos_vacias"]].sum()
    md = [f"# Reporte de actividad - camara {location} ({subset})", "",
          f"Periodo de agregacion: {period}. Fotos: {int(total.fotos)}; con fauna: {int(total.fotos_con_fauna)} "
          f"({100 * total.fotos_con_fauna / total.fotos:.1f}%); vacias: {int(total.fotos_vacias)}.", "",
          f"Umbral de confianza: {m['conf_threshold']} (prioritarias: {m['priority_conf_threshold']}).", "", gt_line, "",
          "## Resumen por periodo", "", summ.to_markdown(index=False) if hasattr(summ, "to_markdown") else summ.to_string(index=False),
          "", "## Actividad por especie y periodo", "", act.to_markdown(index=False) if hasattr(act, "to_markdown") else act.to_string(index=False)]
    try:
        text = "\n".join(md)
    except Exception:
        text = "\n".join(str(x) for x in md)
    (out / f"loc{location}_report.md").write_text(text, encoding="utf-8")

    if len(act):
        piv = act.pivot_table(index="periodo", columns="species", values="individuos", aggfunc="sum").fillna(0)
        fig, ax = plt.subplots(figsize=(8, 3.8))
        piv.plot(kind="bar", stacked=True, ax=ax, width=0.85)
        ax.set_ylabel("individuos detectados"); ax.set_xlabel(period)
        ax.set_title(f"Actividad por especie - camara {location}", fontsize=10)
        ax.legend(fontsize=7, ncol=2); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(out / f"loc{location}_activity_{period}.png", dpi=200); plt.close(fig)
    print(summ.to_string(index=False)); print(act.head(20).to_string(index=False)); print(gt_line)
    print("Guardado en", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--location", default=None)
    ap.add_argument("--subset", default="test_unseen")
    ap.add_argument("--period", default=None, choices=list(FREQ))
    args, cfg = parse_with_config(ap)
    run(cfg, args.weights, args.location, args.subset, args.period)


if __name__ == "__main__":
    main()
