"""Regenera notebooks/Colab_Pipeline.ipynb  (python scripts/make_notebook.py)"""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(t): C.append(nbf.v4.new_markdown_cell(t))
def code(t): C.append(nbf.v4.new_code_cell(t))

md("""# Pregunta 3 - Monitoreo de fauna con camaras trampa (YOLO)
Ejecuta las celdas **en orden**. Runtime > Change runtime type > **GPU (T4)**.
Todo lo pesado (descarga, entrenamiento) corre aqui; los resultados se guardan en tu Google Drive para no perderlos si Colab se desconecta.""")

md("## 0. GPU y Drive")
code("""!nvidia-smi -L
from google.colab import drive
drive.mount('/content/drive')
import os
os.makedirs('/content/drive/MyDrive/cct_yolo_out', exist_ok=True)""")

md("""## 1. Codigo del repo
Opcion A (recomendada): clona tu repo de GitHub. Opcion B: sube `cct_yolo.zip` a `MyDrive/` y se descomprime.""")
code("""REPO_URL = 'https://github.com/SalvadorCT/cct-yolo.git'   # deja '' para usar el zip de Drive
import os, subprocess
os.chdir('/content')
if REPO_URL:
    !git clone {REPO_URL} cct_yolo
else:
    !unzip -q -o /content/drive/MyDrive/cct_yolo.zip -d /content/
%cd /content/cct_yolo
os.environ['PYTHONPATH'] = '/content/cct_yolo/src'   # para que funcione `python -m cct.*`
!pip install -q -r requirements.txt""")

md("""## 2. Descarga de datos (LILA BC)
`metadata` pesa poco. `images` baja ~6 GB (CCT20, imagenes a 1024 px) y tarda un rato.""")
code("""!python -m cct.download --what metadata
!python -m cct.download --what images
!ls data/raw | head""")

md("""## 3. Revisar el JSON de splits
Confirma que `eccv_train` y `eccv_val` aparecen y son listas de ubicaciones (si la estructura es otra, el constructor la imprime).""")
code("""import json
d = json.load(open('data/raw/CaltechCameraTrapsSplits_v0.json'))
def brief(n, depth=0):
    if isinstance(n, dict): return {k: brief(v, depth+1) for k, v in list(n.items())[:12]} if depth < 3 else '{...}'
    if isinstance(n, list): return f'list[{len(n)}] {n[:5]}'
    return n
print(json.dumps(brief(d), indent=1)[:1500])""")

md("## 4. Construir el dataset YOLO (train / val / test_seen / test_unseen)")
code("""!python -m cct.build_dataset --config configs/config.yaml
!cat data/yolo/stats.json | head -40""")
md("Si `stats.json` muestra muy pocas imagenes vacias, descarga extras: `!python -m cct.download --what metadata --fetch-empties 3000` y repite el paso 4.")

md("""## 5. Entrenamiento (entregable 1)
~60 epocas con YOLOv8s a 640 px. Los pesos y `decisions.json` se guardan en Drive. Si Colab se corta, repite los pasos 0-4 (los datos se vuelven a bajar) y reanuda con `--set train.model=<ruta a last.pt en Drive>` (arranca desde esos pesos con el mismo nombre de corrida).""")
code("""RUNS = '/content/drive/MyDrive/cct_yolo_out/runs'
!python -m cct.train --config configs/config.yaml --set paths.runs={RUNS} train.epochs=30 train.workers=4 train.batch=32""")

md("""### Experimento de ablacion de aumentos (opcional, para el informe)
Mismo modelo con aumentos minimos, para justificar la estrategia elegida.""")
code("""!python -m cct.train --config configs/config.yaml --set paths.runs={RUNS} train.aug_preset=baseline train.name=yolov8s_640_baseline train.epochs=30 train.workers=4 train.batch=32""")

md("## 6. Evaluacion por especie (entregable 2)")
code("""W = f'{RUNS}/yolov8s_640_strong/weights/best.pt'
REP = '/content/drive/MyDrive/cct_yolo_out/reports'
!python -m cct.evaluate --config configs/config.yaml --weights {W} --set paths.reports={REP} paths.runs={RUNS}
import pandas as pd
display(pd.read_csv(f'{REP}/tables/summary.csv'))""")

md("## 7. Generalizacion: ubicaciones vistas vs nuevas + Grad-CAM (entregable 3)")
code("""!python -m cct.generalization --config configs/config.yaml --weights {W} --set paths.reports={REP} paths.runs={RUNS}
from IPython.display import Image, display
display(Image(f'{REP}/figures/gradcam_seen_vs_unseen.png'))
display(pd.read_csv(f'{REP}/tables/generalization_drop.csv'))
display(pd.read_csv(f'{REP}/tables/generalization_gradcam_summary.csv'))""")

md("""## 8. Robustez: oclusion y tamano (entregable 4)
Antes (opcional pero recomendado) valida el proxy de oclusion con etiquetas manuales: genera la plantilla, rellena ~60 filas (1 = visible, 2 = parcial, 3 = muy tapado) y corre `--validate`.""")
code("""!python -m cct.occlusion --config configs/config.yaml --make-template 60 --set paths.reports={REP}
# -> descarga/edita {REP}/occlusion_manual_template.csv, vuelve a subirlo y luego:
# !python -m cct.occlusion --config configs/config.yaml --validate --set paths.reports={REP}""")
code("""!python -m cct.robustness --config configs/config.yaml --weights {W} --set paths.reports={REP} paths.runs={RUNS}
display(pd.read_csv(f'{REP}/tables/robustness_drop.csv'))
display(Image(f'{REP}/figures/robustness_groups.png'))""")

md("## 9. Modulo de monitoreo ecologico, demo en una camara no vista (entregable 5)")
code("""!python -m cct.monitoring --config configs/config.yaml --weights {W} --period week --set paths.reports={REP} paths.runs={RUNS}
import glob
for f in glob.glob(f'{REP}/monitoring/*_summary_week.csv'): display(pd.read_csv(f))
for f in glob.glob(f'{REP}/monitoring/*_activity_week.png'): display(Image(f))""")

md("## 10. Empaquetar resultados\nTodo ya esta en Drive (`cct_yolo_out/`). Descarga `reports/` para pegar tablas y figuras en el informe.")
code("""!cd /content/drive/MyDrive/cct_yolo_out && zip -qr reports.zip reports && ls -la reports.zip""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["accelerator"] = "GPU"
Path("notebooks").mkdir(exist_ok=True)
nbf.write(nb, "notebooks/Colab_Pipeline.ipynb")
print("ok")
