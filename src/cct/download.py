"""Descarga de Caltech Camera Traps (LILA BC).

Baja el subconjunto de referencia CCT20 (imagenes reducidas a 1024 px, ~6 GB) y los
metadatos. Si el subconjunto no trae suficientes imagenes vacias, `--fetch-empties N`
baja N imagenes vacias extra, una por una, desde el almacenamiento publico de LILA
(sin el zip gigante de 105 GB) y las reduce a 1024 px.

Uso (Colab):  python -m cct.download --config configs/config.yaml --what all
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

from .utils import add_common_args, data_path, ensure_dir, p, parse_with_config

BASE = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps"
URLS = {
    "images": f"{BASE}/eccv_18_all_images_sm.tar.gz",                 # ~6 GB
    "eccv_annotations": f"{BASE}/eccv_18_annotations.tar.gz",         # ~3 MB
    "bboxes": f"{BASE}/labels/caltech_bboxes_20200316.json",          # ~35 MB
    "image_json": f"{BASE}/labels/caltech_camera_traps.json.zip",     # ~9 MB
    "splits": f"{BASE}/CaltechCameraTrapsSplits_v0.json",             # ~4 KB
}
EMPTY_IMG_URL = "https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/{file_name}"


def _download(url: str, dest: Path) -> None:
    """curl con reanudacion (-C -). Falla con un mensaje claro."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  ya existe: {dest.name}")
        return
    print(f"  bajando {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    cmd = ["curl", "-L", "--fail", "-C", "-", "--retry", "5", "-o", str(tmp), url]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"curl fallo ({r.returncode}) para {url}. Prueba de nuevo: reanuda donde quedo.")
    tmp.rename(dest)


def _extract_tar(archive: Path, out: Path) -> None:
    marker = out / f".extracted_{archive.name}"
    if marker.exists():
        print(f"  ya extraido: {archive.name}")
        return
    print(f"  extrayendo {archive.name} -> {out}")
    ensure_dir(out)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(out)
    marker.touch()


def fetch_empties(cfg: dict, n: int, workers: int = 16) -> int:
    """Baja n imagenes vacias adicionales usando el JSON a nivel de imagen. Devuelve cuantas bajo."""
    import cv2
    import numpy as np
    import requests

    image_json = data_path(cfg, "image_json")
    if not image_json.exists():
        raise FileNotFoundError(f"Falta {image_json}. Corre --what metadata primero.")
    with open(image_json, "r", encoding="utf-8") as f:
        d = json.load(f)
    cats = {c["id"]: c["name"] for c in d["categories"]}
    empty_ids = {c for c, name in cats.items() if name.lower() == "empty"}
    img_by_id = {im["id"]: im for im in d["images"]}
    empties = [img_by_id[a["image_id"]] for a in d["annotations"]
               if a["category_id"] in empty_ids and a["image_id"] in img_by_id]
    # ubicaciones que nos interesan: las de los splits (se filtran de nuevo al construir)
    rng = np.random.default_rng(cfg.get("seed", 42))
    rng.shuffle(empties)
    out = ensure_dir(data_path(cfg, "images_dir") / "extra_empties")
    todo = []
    for im in empties:
        dest = out / (Path(im["file_name"]).stem + ".jpg")
        if not dest.exists():
            todo.append((im, dest))
        if len(todo) >= n:
            break

    def job(item):
        im, dest = item
        try:
            r = requests.get(EMPTY_IMG_URL.format(file_name=im["file_name"]), timeout=60)
            r.raise_for_status()
            arr = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            h, w = arr.shape[:2]
            s = 1024 / max(h, w)
            if s < 1:
                arr = cv2.resize(arr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dest), arr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return True
        except Exception as e:  # una imagen mala no debe detener todo
            print("  fallo", im["file_name"], e)
            return False

    ok = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        for res in ex.map(job, todo):
            ok += int(res)
    print(f"  vacias extra descargadas: {ok}/{len(todo)}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--what", choices=["all", "metadata", "images"], default="all")
    ap.add_argument("--fetch-empties", type=int, default=0, help="N imagenes vacias extra (opcional)")
    args, cfg = parse_with_config(ap)

    raw = ensure_dir(p(cfg, "raw"))
    if args.what in ("all", "metadata"):
        print("Metadatos...")
        _download(URLS["bboxes"], data_path(cfg, "bbox_json"))
        _download(URLS["splits"], data_path(cfg, "splits_json"))
        z = raw / "caltech_camera_traps.json.zip"
        _download(URLS["image_json"], z)
        if not data_path(cfg, "image_json").exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(raw)
        _download(URLS["eccv_annotations"], raw / "eccv_18_annotations.tar.gz")
        _extract_tar(raw / "eccv_18_annotations.tar.gz", raw / "eccv_18_annotations")
    if args.what in ("all", "images"):
        print("Imagenes CCT20 (~6 GB, puede tardar)...")
        _download(URLS["images"], raw / "eccv_18_all_images_sm.tar.gz")
        _extract_tar(raw / "eccv_18_all_images_sm.tar.gz", data_path(cfg, "images_dir"))
    if args.fetch_empties:
        fetch_empties(cfg, args.fetch_empties)
    print("Listo.")


if __name__ == "__main__":
    main()
