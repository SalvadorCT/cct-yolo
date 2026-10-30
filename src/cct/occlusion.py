"""Proxy de oclusion para cajas GT (CCT20 no trae etiquetas de oclusion).

Idea: en una camara trampa el fondo es casi fijo. Para cada ubicacion se estima un fondo
(mediana de fotos vacias de esa camara, separando color de infrarrojo). Dentro de la caja de un
animal medimos que fraccion de pixeles difiere del fondo ("cobertura de primer plano").
Un animal tapado por vegetacion, o camuflado, deja poca cobertura visible.

    visibilidad = min(1, cobertura / referencia_de_la_clase)     (referencia = percentil 90 de la clase)
    oclusion    = 1 - visibilidad                                (se parte en terciles: baja/media/alta)

Es un PROXY, no una medida directa. Por eso se valida con etiquetas manuales:
  python -m cct.occlusion --make-template N   -> reports/occlusion_manual_template.csv (rellenar 1/2/3)
  python -m cct.occlusion --validate          -> Spearman proxy vs manual
Ademas se marca `trunc` si la caja toca el borde (animal cortado por el encuadre).
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import pandas as pd

from .utils import add_common_args, ensure_dir, p, parse_with_config, read_manifest, set_seed


def _load_small(path: str, width: int):
    im = cv2.imread(path)
    if im is None:
        return None
    h, w = im.shape[:2]
    s = width / w
    return cv2.resize(im, (width, max(1, int(round(h * s)))), interpolation=cv2.INTER_AREA)


def _is_ir(bgr: np.ndarray) -> bool:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[..., 1].mean()) < 12.0


def _gray_norm(bgr: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ref = g[mask] if mask is not None and mask.any() else g.ravel()
    med = np.median(ref)
    sd = 1.4826 * np.median(np.abs(ref - med)) + 1e-3
    return (g - med) / max(sd, 2.0)


def build_backgrounds(manifest: pd.DataFrame, locations, width: int, n_samples: int, seed: int) -> dict:
    """{(location, 'color'|'ir'): imagen gris normalizada}. Fotos vacias; si faltan, cualquier foto."""
    rng = np.random.default_rng(seed)
    bg = {}
    for loc in locations:
        rows = manifest[manifest.location == loc]
        for pool in (rows[rows.is_empty == 1], rows):
            frames = {"color": [], "ir": []}
            paths = pool["image_path"].tolist()
            rng.shuffle(paths)
            for pth in paths[: n_samples * 2]:
                im = _load_small(pth, width)
                if im is None:
                    continue
                frames["ir" if _is_ir(im) else "color"].append(im)
            if sum(len(v) for v in frames.values()) >= 3:
                break
        allf = frames["color"] + frames["ir"]
        for mode in ("color", "ir"):
            fr = frames[mode] if len(frames[mode]) >= 3 else allf
            if len(fr) < 1:
                continue
            shape = fr[0].shape[:2]
            fr = [cv2.resize(f, (shape[1], shape[0])) for f in fr[:n_samples]]
            med = np.median(np.stack(fr), axis=0).astype(np.uint8)
            bg[(loc, mode)] = _gray_norm(med)
    return bg


def box_coverage(manifest_row, boxes: np.ndarray, bg: dict, width: int) -> list[float]:
    """Cobertura de primer plano dentro de cada caja (xyxy normalizado) de una imagen."""
    im = _load_small(manifest_row.image_path, width)
    if im is None:
        return [np.nan] * len(boxes)
    mode = "ir" if _is_ir(im) else "color"
    b = next((bg[k] for k in ((manifest_row.location, mode), (manifest_row.location, "color"),
                              (manifest_row.location, "ir")) if k in bg), None)
    if b is None:
        return [np.nan] * len(boxes)
    H, W = im.shape[:2]
    b = cv2.resize(b, (W, H))
    inbox = np.zeros((H, W), bool)
    for x0, y0, x1, y1 in boxes:
        inbox[int(y0 * H):int(y1 * H) + 1, int(x0 * W):int(x1 * W) + 1] = True
    g = _gray_norm(im, ~inbox)
    d = np.abs(g - b)
    out_d = d[~inbox] if (~inbox).any() else d.ravel()
    thr = max(float(np.median(out_d) + 3 * 1.4826 * np.median(np.abs(out_d - np.median(out_d)))), 0.75)
    cov = []
    for x0, y0, x1, y1 in boxes:
        sl = d[int(y0 * H):int(y1 * H) + 1, int(x0 * W):int(x1 * W) + 1]
        cov.append(float((sl > thr).mean()) if sl.size else np.nan)
    return cov


def compute(cfg: dict, gt: pd.DataFrame, manifest: pd.DataFrame, cache: str | None = None) -> pd.DataFrame:
    """Devuelve gt con columnas cov, occ (0-1), trunc, occ_group (baja/media/alta). Mismo orden que gt."""
    if cache:
        from pathlib import Path
        cp = Path(cache)
        if cp.exists():
            c = pd.read_csv(cp)
            if len(c) == len(gt):
                return c
    r = cfg["robustness"]
    set_seed(cfg["seed"])
    bg = build_backgrounds(manifest, gt["location"].unique(), r["proxy_width"], r["bg_samples_per_location"], cfg["seed"])
    mrow = {x.name: x for x in manifest.itertuples()}
    cov = np.full(len(gt), np.nan)
    gtr = gt.reset_index(drop=True)
    for name, idx in gtr.groupby("name").groups.items():
        idx = np.array(list(idx))
        cov[idx] = box_coverage(mrow[name], gtr.loc[idx, ["xmin", "ymin", "xmax", "ymax"]].to_numpy(), bg, r["proxy_width"])
    out = gtr.copy()
    out["cov"] = cov
    ref = out.groupby("cls")["cov"].transform(lambda s: s.quantile(0.9) if s.notna().sum() >= 10 else np.nan)
    ref = ref.fillna(out["cov"].quantile(0.9)).clip(lower=1e-3)
    out["occ"] = 1 - np.minimum(1.0, out["cov"] / ref)
    eps = 0.01
    out["trunc"] = ((out.xmin <= eps) | (out.ymin <= eps) | (out.xmax >= 1 - eps) | (out.ymax >= 1 - eps)).astype(int)
    qs = out["occ"].quantile([1 / 3, 2 / 3]).to_numpy()
    out["occ_group"] = np.where(out.occ <= qs[0], "baja", np.where(out.occ <= qs[1], "media", "alta"))
    out.loc[out["occ"].isna(), "occ_group"] = "desconocida"
    if cache:
        ensure_dir(__import__("pathlib").Path(cache).parent)
        out.to_csv(cache, index=False)
    return out


def make_template(cfg: dict, n: int) -> None:
    from .eval_utils import load_gt
    m = read_manifest(cfg)
    gt = load_gt(m[m.subset.isin(["test_seen", "test_unseen"]) & (m.is_empty == 0)])
    s = gt.sample(min(n, len(gt)), random_state=cfg["seed"])[["name", "cls", "xmin", "ymin", "xmax", "ymax"]].copy()
    s["manual_occlusion_1_2_3"] = ""   # 1 = animal visible, 2 = parcialmente tapado, 3 = muy tapado
    out = ensure_dir(p(cfg, "reports")) / "occlusion_manual_template.csv"
    s.to_csv(out, index=False)
    print(f"Plantilla: {out} -> rellena la ultima columna (1/2/3) y corre --validate")


def validate(cfg: dict, path: str | None = None) -> None:
    from scipy.stats import spearmanr
    from .eval_utils import load_gt
    path = path or str(p(cfg, "reports") / "occlusion_manual_template.csv")
    man = pd.read_csv(path).dropna(subset=["manual_occlusion_1_2_3"])
    m = read_manifest(cfg)
    gt = load_gt(m[m.subset.isin(["test_seen", "test_unseen"]) & (m.is_empty == 0)])
    occ = compute(cfg, gt, m)
    mg = man.merge(occ, on=["name", "cls", "xmin", "ymin", "xmax", "ymax"], how="inner", suffixes=("", "_p"))
    rho, pv = spearmanr(mg["manual_occlusion_1_2_3"], mg["occ"], nan_policy="omit")
    res = pd.DataFrame([dict(n=len(mg), spearman=rho, p_value=pv)])
    res.to_csv(ensure_dir(p(cfg, "reports") / "tables") / "occlusion_validation.csv", index=False)
    print(res.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--make-template", type=int, default=0)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--manual-csv", default=None)
    args, cfg = parse_with_config(ap)
    if args.make_template:
        make_template(cfg, args.make_template)
    if args.validate:
        validate(cfg, args.manual_csv)


if __name__ == "__main__":
    main()
