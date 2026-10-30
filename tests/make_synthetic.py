"""Genera un mini-dataset sintetico con el mismo formato que Caltech Camera Traps (COCO-CT).
Sirve solo para probar que el pipeline corre de principio a fin sin descargar 6 GB."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

CLASSES = {1: "opossum", 2: "raccoon", 3: "deer"}   # id 0 = empty
COLORS = {1: (40, 40, 220), 2: (40, 200, 40), 3: (220, 60, 60)}


def make(root: Path, n_train_loc=4, n_val_loc=2, seqs=6, frames=3, seed=0):
    rng = np.random.default_rng(seed)
    raw = root / "data" / "raw"
    imgdir = raw / "images" / "cct_images"
    imgdir.mkdir(parents=True, exist_ok=True)
    locs = [str(10 + i) for i in range(n_train_loc + n_val_loc)]
    train_locs, val_locs = locs[:n_train_loc], locs[n_train_loc:]
    cats = [{"id": 0, "name": "empty"}] + [{"id": i, "name": n} for i, n in CLASSES.items()]
    images, anns, k = [], [], 0
    W, H = 160, 120
    for loc in locs:
        base = rng.integers(60, 180, 3)
        tex = (rng.normal(0, 12, (H, W, 3)) + base).clip(0, 255).astype(np.uint8)
        tex = cv2.GaussianBlur(tex, (0, 0), 3)
        for s in range(seqs):
            seq_id = f"{loc}_{s}"
            for f in range(frames):
                img = np.clip(tex + rng.normal(0, 4, tex.shape), 0, 255).astype(np.uint8)
                empty = rng.random() < 0.3
                iid = f"img{k:05d}"
                fn = f"cct_images/{iid}.jpg"
                dt = f"2013-0{1 + (s % 3)}-{10 + s:02d} {8 + f * 4:02d}:00:00"
                images.append(dict(id=iid, file_name=fn, width=W, height=H, location=loc, seq_id=seq_id,
                                   date_captured=dt, seq_num_frames=frames, frame_num=f))
                if not empty:
                    for _ in range(int(rng.integers(1, 3))):
                        c = int(rng.integers(1, 4))
                        w, h = int(rng.integers(25, 60)), int(rng.integers(20, 50))
                        x, y = int(rng.integers(0, W - w)), int(rng.integers(0, H - h))
                        img[y:y + h, x:x + w] = COLORS[c]
                        anns.append(dict(id=f"a{len(anns)}", image_id=iid, category_id=c, bbox=[x, y, w, h]))
                cv2.imwrite(str(imgdir / f"{iid}.jpg"), img)
                k += 1
    boxed = {a["image_id"] for a in anns}
    bb = dict(images=[i for i in images if i["id"] in boxed], annotations=anns, categories=cats)
    ij_anns = [dict(id=f"i{n}", image_id=i["id"], category_id=(0 if i["id"] not in boxed else 1)) for n, i in enumerate(images)]
    ij = dict(images=images, annotations=ij_anns, categories=cats)
    (raw / "caltech_bboxes_20200316.json").write_text(json.dumps(bb))
    (raw / "caltech_camera_traps.json").write_text(json.dumps(ij))
    (raw / "CaltechCameraTrapsSplits_v0.json").write_text(json.dumps(
        {"splits": {"train": train_locs, "val": [], "cis_val": [], "cis_test": [], "trans_val": [], "trans_test": [],
                    "eccv_train": train_locs, "eccv_val": val_locs}}))
    print("Sintetico en", raw)


if __name__ == "__main__":
    make(Path(sys.argv[1]))
