"""Genera las tablas LaTeX del informe a partir de results/tables/*.csv.
Uso (desde la raiz del repo):  python scripts/make_report_tables.py
Salida: report/tablas/*.tex  (se incluyen con \\input en report/main.tex)"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
OUT = ROOT / "report" / "tablas"
OUT.mkdir(parents=True, exist_ok=True)


def f(x, d=3):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def write(name, header, rows, spec):
    lines = [r"\begin{tabular}{" + spec + "}", r"\hline", " & ".join(header) + r" \\", r"\hline"]
    lines += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    lines += [r"\hline", r"\end{tabular}"]
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- resumen global ---
s = pd.read_csv(T / "summary.csv").set_index("subset")
g = pd.read_csv(T / "generalization_gap.csv").set_index("subset")
rows = []
for k, lab in (("test_seen", "Ubicaciones vistas"), ("test_unseen", "Ubicaciones nuevas")):
    rows.append([lab, f"{int(s.loc[k, 'n_images']):,}", f"{int(s.loc[k, 'n_boxes']):,}", f(s.loc[k, "map50"]), f(s.loc[k, "map5095"]),
                 f(g.loc[k, "recall_at_conf"]), f(s.loc[k, "frac_with_fp"], 2)])
write("tab_resumen.tex", ["Conjunto", "Im\\'agenes", "Cajas", "mAP@0.5", "mAP@0.5:0.95", "Recall", "Vac\\'ias con FP"], rows, "lrrcccc")

# --- por especie ---
a = pd.read_csv(T / "per_class_test_seen.csv").set_index("especie")
b = pd.read_csv(T / "per_class_test_unseen.csv").set_index("especie")
order = a.sort_values("n_train", ascending=False).index
rows = []
for e in order:
    rows.append([e.replace("_", " "), f"{int(a.loc[e, 'n_train']):,}", f(a.loc[e, "ap50"]), f(b.loc[e, "ap50"]),
                 f(a.loc[e, "ap5095"]), f(b.loc[e, "ap5095"]), f"{int(b.loc[e, 'n_gt']):,}"])
write("tab_porclase.tex", ["Especie", "Inst. train", "AP50 vistas", "AP50 nuevas", "AP50-95 vistas", "AP50-95 nuevas", "Cajas nuevas"],
      rows, "lrcccrr")

# --- comunes vs raras ---
cs, cu = pd.read_csv(T / "common_vs_rare_test_seen.csv").set_index("grupo"), pd.read_csv(T / "common_vs_rare_test_unseen.csv").set_index("grupo")
rows = []
for grp in ("comun", "rara"):
    rows.append([{"comun": "Comunes", "rara": "Raras"}[grp], f(cs.loc[grp, "ap50"]), f(cu.loc[grp, "ap50"]), f(cs.loc[grp, "ap5095"]), f(cu.loc[grp, "ap5095"])])
rows.append(["Brecha", f(cs.loc["comun", "ap50"] - cs.loc["rara", "ap50"]), f(cu.loc["comun", "ap50"] - cu.loc["rara", "ap50"]),
             f(cs.loc["comun", "ap5095"] - cs.loc["rara", "ap5095"]), f(cu.loc["comun", "ap5095"] - cu.loc["rara", "ap5095"])])
write("tab_comun_rara.tex", ["Grupo", "AP50 vistas", "AP50 nuevas", "AP50-95 vistas", "AP50-95 nuevas"], rows, "lcccc")

# --- Grad-CAM ---
gc = pd.read_csv(T / "generalization_gradcam_summary.csv").set_index("subset")
rows = []
for k, lab in (("test_seen", "Vistas"), ("test_unseen", "Nuevas")):
    rows.append([lab, int(gc.loc[k, "n"]), f(gc.loc[k, "energia_en_caja"]), f(gc.loc[k, "area_caja"]), f(gc.loc[k, "energia_sobre_area"], 2),
                 f(100 * gc.loc[k, "detecta_original"], 0) + "\\,\\%", f(100 * gc.loc[k, "detecta_tras_enmascarar"], 1) + "\\,\\%"])
write("tab_gradcam.tex", ["Ubicaciones", "$n$", "Energ\\'ia en caja", "\\'Area de la caja", "Cociente", "Detecta", "Detecta tras tapar"], rows, "lrccccc")

# --- robustez ---
rows = []
for fac, files, labs in (("Tama\\~no", "size", [("pequena", "peque\\~na"), ("mediana", "mediana"), ("grande", "grande")]),
                         ("Oclusi\\'on (proxy)", "occlusion", [("baja", "baja"), ("media", "media"), ("alta", "alta")]),
                         ("Truncamiento", "truncation", [("completo", "completa"), ("cortado", "cortada")])):
    ds = {k: pd.read_csv(T / f"robustness_{files}_{k}.csv").set_index("grupo") for k in ("test_seen", "test_unseen")}
    for i, (key, lab) in enumerate(labs):
        rows.append([fac if i == 0 else "", lab] + [x for k in ("test_seen", "test_unseen")
                     for x in (f(ds[k].loc[key, "map50"]), f(ds[k].loc[key, "map5095"]), f(ds[k].loc[key, "recall_conf"]))])
write("tab_robustez.tex", ["Factor", "Grupo", "mAP50", "mAP50-95", "Rec.", "mAP50", "mAP50-95", "Rec."], rows, "llcccccc")

# --- monitoreo: conteos verdad vs deteccion ---
m = ROOT / "results" / "monitoring"
cmp_path = m / "loc100_gt_vs_pred.csv"
if cmp_path.exists():
    c = pd.read_csv(cmp_path).sort_values("gt_cajas", ascending=False)
    rows = [[r.especie, int(r.gt_cajas), int(r.detecciones), int(r.individuos)] for r in c.itertuples()]
    write("tab_monitoreo.tex", ["Especie", "Cajas reales", "Detecciones", "Individuos"], rows, "lrrr")
print("tablas escritas en", OUT)
