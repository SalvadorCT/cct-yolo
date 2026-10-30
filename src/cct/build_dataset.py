"""Convierte Caltech Camera Traps (formato COCO Camera Traps) a formato YOLO.

Decisiones que quedan documentadas (entregable 1 y 3):
  * `eccv_val`  (ubicaciones "trans" del paper ECCV18)  -> test_unseen: camaras NUNCA vistas.
  * `eccv_train` (ubicaciones "train" + "cis")           -> se divide POR SECUENCIA en
    train / val / test_seen. Dividir por secuencia (no por imagen) evita que fotogramas casi
    identicos del mismo disparo caigan en conjuntos distintos (fuga de informacion).
  * Las imagenes vacias NO son una clase: son imagenes de fondo (archivo de etiquetas vacio).
    Se submuestrean (train_empty_ratio / eval_empty_ratio) porque ~70 % del dataset esta vacio.
  * Desbalance de especies: repeat-factor sampling opcional (Gupta et al., LVIS 2019) solo en train.

Uso:  python -m cct.build_dataset --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .utils import add_common_args, data_path, ensure_dir, p, parse_with_config, set_seed

IMG_EXT = {".jpg", ".jpeg", ".png"}


# ----------------------------------------------------------------------------- lectura
def index_images(images_dir: Path) -> dict[str, Path]:
    """basename -> ruta, buscando recursivamente."""
    idx: dict[str, Path] = {}
    for path in images_dir.rglob("*"):
        if path.suffix.lower() in IMG_EXT:
            idx.setdefault(path.name, path)
    return idx


def _walk(node, key_path=()):
    yield key_path, node
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, key_path + (k,))


def load_location_splits(path: Path, train_key: str, val_key: str) -> tuple[set[str], set[str], str]:
    """Lee CaltechCameraTrapsSplits_v0.json de forma tolerante.

    Soporta (a) listas bajo claves `eccv_train` / `eccv_val` en cualquier nivel y
    (b) un diccionario ubicacion -> nombre_de_split. Si no encuentra nada, imprime la
    estructura para que sepas como esta armado el archivo.
    """
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    found: dict[str, set[str]] = {}
    for kp, node in _walk(d):
        if kp and kp[-1] in (train_key, val_key) and isinstance(node, list):
            found[kp[-1]] = {str(x) for x in node}
    if train_key in found and val_key in found:
        return found[train_key], found[val_key], "listas"
    for kp, node in _walk(d):
        if isinstance(node, dict) and node and all(isinstance(v, str) for v in node.values()):
            vals = set(node.values())
            if train_key in vals and val_key in vals:
                tr = {str(k) for k, v in node.items() if v == train_key}
                va = {str(k) for k, v in node.items() if v == val_key}
                return tr, va, "mapa ubicacion->split"
    def brief(n, depth=0):
        if depth > 2 or not isinstance(n, dict):
            return type(n).__name__
        return {k: brief(v, depth + 1) for k, v in list(n.items())[:8]}
    raise KeyError(f"No encontre '{train_key}'/'{val_key}' en {path.name}. Estructura: {json.dumps(brief(d), indent=1)[:1500]}\n"
                   "Ajusta data.train_split_key / data.val_split_key en configs/config.yaml.")


def load_coco_ct(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _stable_u(seed: int, key: str) -> float:
    h = hashlib.md5(f"{seed}-{key}".encode()).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))


def _meta_time(im: dict) -> str:
    return str(im.get("datetime") or im.get("date_captured") or "")


# ----------------------------------------------------------------------------- principal
def build(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    dcfg = cfg["data"]
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)
    out = ensure_dir(p(cfg, "yolo"))

    print("Leyendo anotaciones...")
    bb = load_coco_ct(data_path(cfg, "bbox_json"))
    cats = {c["id"]: c["name"] for c in bb["categories"]}
    empty_cat_ids = {i for i, n in cats.items() if n.lower() == "empty"}
    drop = {n.lower() for n in dcfg.get("drop_classes", [])}
    images = {im["id"]: im for im in bb["images"]}
    anns = defaultdict(list)
    for a in bb["annotations"]:
        if "bbox" not in a or a["category_id"] in empty_cat_ids:
            continue
        if cats.get(a["category_id"], "").lower() in drop:
            continue
        anns[a["image_id"]].append(a)

    # --- imagenes vacias confirmadas ---
    empties: dict[str, dict] = {}
    image_json = data_path(cfg, "image_json")
    if image_json.exists():
        ij = load_coco_ct(image_json)
        icats = {c["id"]: c["name"] for c in ij["categories"]}
        iimg = {im["id"]: im for im in ij["images"]}
        for a in ij["annotations"]:
            if icats.get(a["category_id"], "").lower() == "empty" and a["image_id"] in iimg:
                empties[a["image_id"]] = iimg[a["image_id"]]
        nonempty_ids = {a["image_id"] for a in ij["annotations"]
                        if icats.get(a["category_id"], "").lower() != "empty"}
    else:
        print("AVISO: no hay image_json; se asume vacia toda imagen del JSON de cajas sin anotaciones.")
        nonempty_ids = set()
        for iid, im in images.items():
            if iid not in anns:
                empties[iid] = im
    for iid, im in images.items():  # imagenes sin cajas que ademas no son animales segun el JSON a nivel de imagen
        if iid not in anns and iid in empties:
            empties[iid] = im

    # --- splits por ubicacion ---
    tr_locs, va_locs, how = load_location_splits(data_path(cfg, "splits_json"),
                                                 dcfg["train_split_key"], dcfg["val_split_key"])
    print(f"Splits ({how}): {len(tr_locs)} ubicaciones eccv_train, {len(va_locs)} eccv_val")

    # --- archivos en disco ---
    idx = index_images(data_path(cfg, "images_dir"))
    print(f"Imagenes encontradas en disco: {len(idx)}")
    if not idx:
        raise FileNotFoundError("No hay imagenes bajo data.images_dir. Corre primero cct.download.")

    def resolve(im):
        return idx.get(Path(im["file_name"]).name)

    animal_ids = [i for i in anns if i in images]
    sample_locs = {str(images[i].get("location")) for i in animal_ids[:2000]}
    if not (sample_locs & (tr_locs | va_locs)):
        raise ValueError("Ninguna ubicacion de las imagenes coincide con los splits.\n"
                         f"  ubicaciones en imagenes (muestra): {sorted(sample_locs)[:10]}\n"
                         f"  ubicaciones en splits (muestra): {sorted(tr_locs | va_locs)[:10]}")

    def subset_for(im, kind: str):
        loc = str(im.get("location"))
        if loc in va_locs:
            return "test_unseen"
        if loc in tr_locs:
            u = _stable_u(seed, str(im.get("seq_id", im["id"])))
            if u < dcfg["holdout_test_seen_frac"]:
                return "test_seen"
            if u < dcfg["holdout_test_seen_frac"] + dcfg["holdout_val_frac"]:
                return "val"
            return "train"
        return None

    animals_by_subset: dict[str, list] = defaultdict(list)
    skipped_nofile = 0
    for iid in animal_ids:
        im = images[iid]
        s = subset_for(im, "animal")
        if s is None:
            continue
        if resolve(im) is None:
            skipped_nofile += 1
            continue
        animals_by_subset[s].append(im)

    empties_by_subset: dict[str, list] = defaultdict(list)
    for iid, im in empties.items():
        s = subset_for(im, "empty")
        if s is None or resolve(im) is None:
            continue
        empties_by_subset[s].append(im)
    # extras descargadas con cct.download --fetch-empties (viven en images_dir/extra_empties)
    print(f"Con animal por subconjunto: { {k: len(v) for k, v in animals_by_subset.items()} } | sin archivo: {skipped_nofile}")
    print(f"Vacias disponibles: { {k: len(v) for k, v in empties_by_subset.items()} }")

    # --- clases ---
    used_cat_ids = sorted({a["category_id"] for s in animals_by_subset.values() for im in s for a in anns[im["id"]]})
    classes = [cats[c] for c in used_cat_ids]
    cat2cls = {c: i for i, c in enumerate(used_cat_ids)}
    print(f"Clases ({len(classes)}): {classes}")

    # --- muestreo de vacias ---
    chosen_empties: dict[str, list] = {}
    for s in ("train", "val", "test_seen", "test_unseen"):
        ratio = dcfg["train_empty_ratio"] if s == "train" else dcfg["eval_empty_ratio"]
        pool = empties_by_subset.get(s, [])
        k = min(len(pool), int(round(ratio * len(animals_by_subset.get(s, [])))))
        order = rng.permutation(len(pool))[:k]
        chosen_empties[s] = [pool[i] for i in order]

    # --- escritura ---
    for sub in ("images", "labels"):
        shutil.rmtree(out / sub, ignore_errors=True)
    rows = []
    class_counts = {s: Counter() for s in ("train", "val", "test_seen", "test_unseen")}
    min_side = float(dcfg.get("min_box_side", 0.0))
    train_paths: list[Path] = []
    img_classes: dict[str, set] = {}

    for s in ("train", "val", "test_seen", "test_unseen"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)
        items = [(im, False) for im in animals_by_subset.get(s, [])] + [(im, True) for im in chosen_empties.get(s, [])]
        for im, is_empty in items:
            src = resolve(im).resolve()
            name = _safe(im["id"]) + ".jpg"
            dst = out / "images" / s / name
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)
            W, H = im.get("width"), im.get("height")
            if not W or not H:
                import cv2
                H, W = cv2.imread(str(src)).shape[:2]
            lines, present = [], set()
            if not is_empty:
                for a in anns[im["id"]]:
                    x, y, w, h = a["bbox"]
                    x0, y0 = max(0.0, x / W), max(0.0, y / H)
                    x1, y1 = min(1.0, (x + w) / W), min(1.0, (y + h) / H)
                    bw, bh = x1 - x0, y1 - y0
                    if bw < min_side or bh < min_side:
                        continue
                    c = cat2cls[a["category_id"]]
                    lines.append(f"{c} {(x0 + x1) / 2:.6f} {(y0 + y1) / 2:.6f} {bw:.6f} {bh:.6f}")
                    class_counts[s][classes[c]] += 1
                    present.add(c)
            (out / "labels" / s / (name[:-4] + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
            rows.append(dict(name=name, subset=s, location=str(im.get("location")), seq_id=str(im.get("seq_id", im["id"])),
                             datetime=_meta_time(im), width=W, height=H, n_boxes=len(lines),
                             is_empty=int(is_empty or len(lines) == 0), image_path=str(dst), orig_path=str(src)))
            if s == "train":
                train_paths.append(dst)
                img_classes[str(dst)] = present

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "manifest.csv", index=False)
    (out / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=1))

    # --- repeat-factor sampling (solo train) ---
    repeats = {}
    if dcfg.get("rare_oversample", False) and train_paths:
        t = float(dcfg["rfs_threshold"])
        n_img = len(train_paths)
        f = Counter()
        for pth in train_paths:
            for c in img_classes[str(pth)]:
                f[c] += 1
        r_c = {c: max(1.0, math.sqrt(t / (f[c] / n_img))) for c in f}
        for pth in train_paths:
            cs = img_classes[str(pth)]
            r = max([r_c[c] for c in cs], default=1.0)
            k = int(math.floor(r)) + int(rng.random() < (r - math.floor(r)))
            repeats[str(pth)] = max(1, k)
    lines = []
    for pth in train_paths:
        lines += [str(pth)] * repeats.get(str(pth), 1)
    (out / "train.txt").write_text("\n".join(lines) + "\n")

    # --- yamls ---
    names = {i: n for i, n in enumerate(classes)}
    def write_yaml(fname, train, val, test=None):
        d = {"path": str(out), "train": train, "val": val, "names": names}
        if test:
            d["test"] = test
        (out / fname).write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
    write_yaml("data.yaml", str(out / "train.txt"), "images/val", "images/test_unseen")
    write_yaml("data_seen.yaml", str(out / "train.txt"), "images/test_seen")
    write_yaml("data_unseen.yaml", str(out / "train.txt"), "images/test_unseen")

    # --- estadisticas ---
    stats = {
        "classes": classes,
        "images": manifest.groupby("subset").size().to_dict(),
        "images_empty": manifest[manifest.is_empty == 1].groupby("subset").size().to_dict(),
        "instances": {s: dict(c) for s, c in class_counts.items()},
        "locations": manifest.groupby("subset")["location"].nunique().to_dict(),
        "train_list_len_with_repeats": len(lines),
        "split_source": {"train_locations": len(tr_locs), "val_locations": len(va_locs)},
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False))
    cc = pd.DataFrame(class_counts).fillna(0).astype(int).reindex(classes).fillna(0).astype(int)
    cc.to_csv(out / "class_counts.csv")
    print("\nResumen:")
    print(json.dumps({k: v for k, v in stats.items() if k != "instances"}, indent=1, ensure_ascii=False))
    print("\nInstancias por clase y subconjunto:")
    print(cc.to_string())
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    args, cfg = parse_with_config(ap)
    build(cfg)


if __name__ == "__main__":
    main()
