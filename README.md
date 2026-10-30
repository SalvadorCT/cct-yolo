# Monitoreo de fauna silvestre con cámaras trampa (YOLOv8) — Caltech Camera Traps

Proyecto de la **Pregunta 3** del examen parcial 2026-II del curso *Redes Neuronales y Aprendizaje Profundo* (Maestría en Inteligencia Artificial, UNI).
Grupo 4: Salvador Cabrera Taco, Anllelo Carreazo Yrizales, Yda Ramirez Rojas, Pedro Alexander Marquez Lamadrid, Mijael Aguirre Zegarra.

## ¿De qué trata?

Las cámaras trampa toman miles de fotos y revisarlas a mano es lento. Entrenamos un detector **YOLOv8s** sobre el conjunto público
[Caltech Camera Traps (CCT-20, LILA BC)](https://lila.science/datasets/caltech-camera-traps) para encontrar animales de 15 categorías, y estudiamos lo que más importa en la práctica:

1. **¿Funciona en cámaras que nunca vio?** (partición oficial `eccv_train` / `eccv_val`).
2. **¿Qué pasa con las especies raras?** (efecto del desbalance).
3. **¿Qué tanto afectan el tamaño del animal, el truncamiento y la oclusión?**
4. **¿Se puede convertir el detector en un reporte de actividad por especie y periodo?**

El informe del trabajo está en `report/main.tex` (se compila en Overleaf).

## Resultados principales

| Conjunto | Imágenes | mAP@0.5 | mAP@0.5:0.95 | Recall (conf. 0.25) |
|---|---:|---:|---:|---:|
| Cámaras vistas (`test_seen`) | 2,784 | 0.841 | 0.593 | 0.936 |
| Cámaras nuevas (`test_unseen`) | 19,497 | 0.462 | 0.314 | 0.739 |

- **Brecha de generalización:** −45.0 % de mAP@0.5 y −47.0 % de mAP@0.5:0.95 al pasar a cámaras nuevas.
- **Desbalance:** AP@0.5 de especies comunes vs. raras = 0.914 vs. 0.696 (cámaras vistas) y 0.637 vs. 0.149 (cámaras nuevas).
- **Robustez (cámaras nuevas):** tamaño pequeño vs. grande: −34.8 % de mAP@0.5:0.95. Cajas cortadas por el borde vs. completas: −58.2 % de mAP@0.5. El proxy de oclusión no mostró caída (no está validado).
- **Grad-CAM:** evidencia mixta sobre el uso del fondo (ver informe). Al tapar al animal, el modelo casi no sigue detectándolo.
- **Monitoreo (cámara 100, no vista):** detecta 10 de las 11 especies presentes; la tasa de fotos vacías del reporte no es fiable.
- **Limitación importante:** casi todas las fotos vacías (94–96 %) reciben alguna detección a confianza 0.25.

Todas las tablas y figuras están en [`results/`](results/) y los números exactos en `results/tables/*.csv`.

## Cómo ejecutarlo paso a paso (Google Colab, recomendado)

No necesitas descargar datos a tu PC: el notebook los baja solo a Colab.

1. Abre Colab y elige **Archivo → Abrir cuaderno → GitHub**, pega `SalvadorCT/cct-yolo` y abre `notebooks/Colab_Pipeline.ipynb`.
2. **Entorno de ejecución → Cambiar tipo de entorno** y elige una GPU (T4 sirve; L4 o A100 si tienes Colab Pro).
3. En la celda 1 escribe tu URL: `REPO_URL = 'https://github.com/SalvadorCT/cct-yolo.git'`.
4. Ejecuta las celdas **en orden, una por una**:
   - 0: verifica la GPU y monta Google Drive (los resultados se guardan en `MyDrive/cct_yolo_out/`).
   - 1: clona el repo e instala las dependencias.
   - 2: descarga CCT-20 desde LILA (~6 GB, tarda un rato).
   - 3–4: revisa el JSON de particiones y construye el dataset en formato YOLO.
   - 5: entrena (30 épocas ≈ 1.2 h en una A100; añade `train.epochs=30 train.workers=4 train.batch=32` al comando).
   - 6: evaluación por especie. 7: generalización y Grad-CAM. 8: robustez. 9: monitoreo. 10: comprime los resultados.
5. Descarga `reports.zip` desde tu Drive. Si algo falla con `No module named 'cct'`, ejecuta:
   ```python
   import os; os.chdir('/content/cct_yolo'); os.environ['PYTHONPATH'] = '/content/cct_yolo/src'
   ```

Cualquier parámetro se cambia sin editar código con `--set clave=valor`, por ejemplo `--set train.epochs=5`.

## Cómo probar el código sin GPU ni descargas

Hay una prueba con datos sintéticos (tarda unos minutos en CPU):

```bash
pip install -r requirements.txt
python tests/test_eval.py      # pruebas unitarias de las métricas y del monitoreo
python tests/smoke_test.py     # pipeline completo con imágenes sintéticas
```

## Estructura del repositorio

```
configs/config.yaml        # todos los parámetros (semilla, rutas, entrenamiento, evaluación, monitoreo)
notebooks/Colab_Pipeline.ipynb   # flujo completo para Colab
src/cct/
  download.py              # descarga de LILA BC
  build_dataset.py         # COCO Camera Traps -> formato YOLO, particiones sin fuga por secuencia
  train.py                 # entrenamiento YOLOv8 y registro de decisiones (decisions.json)
  inference.py, eval_utils.py, evaluate.py   # predicción y métricas (mAP por especie, desbalance)
  gradcam.py, generalization.py              # Grad-CAM propio y análisis cámaras vistas vs. nuevas
  occlusion.py, robustness.py                # proxy de oclusión y robustez por tamaño/oclusión/truncamiento
  monitoring.py            # reporte de actividad por especie y periodo
scripts/                   # descarga, regeneración del notebook y de las tablas del informe
tests/                     # pruebas unitarias y prueba de humo con datos sintéticos
results/                   # tablas (CSV/LaTeX), figuras, reporte de monitoreo y decisions.json de nuestra corrida
report/                    # informe en LaTeX (main.tex, refs.bib, imagenes/, tablas/)
```

## ¿Qué entregable está dónde?

| # | Entregable | Código | Resultados |
|---|---|---|---|
| 1 | YOLO ajustado y decisiones (resolución, aumentos, vacías como fondo) | `train.py`, `build_dataset.py` | `results/decisions.json` |
| 2 | mAP@0.5 y @0.5:0.95 por especie, efecto del desbalance | `evaluate.py` | `results/tables/per_class_*`, `common_vs_rare_*`, `summary.csv` |
| 3 | Generalización eccv_train/eccv_val y Grad-CAM | `generalization.py`, `gradcam.py` | `results/tables/generalization_*`, `results/figures/gradcam_seen_vs_unseen.png` |
| 4 | Robustez por oclusión y tamaño | `occlusion.py`, `robustness.py` | `results/tables/robustness_*` |
| 5 | Módulo de monitoreo ecológico con demo | `monitoring.py` | `results/monitoring/` |
| 6 | Discusión de IA responsable | — | `report/main.tex`, sección 5 |

## Decisiones de diseño

- **Particiones sin fuga:** train/val/`test_seen` se separan por *secuencia* (hash de `seq_id`), no por imagen, porque los cuadros de una ráfaga son casi idénticos. `test_unseen` son las ubicaciones de `eccv_val`.
- **Imágenes vacías como fondo** (archivo de etiquetas vacío) con un tope respecto a las imágenes con animales.
- **Desbalance:** *repeat factor sampling* (Gupta et al., 2019).
- **Resolución 640 px** y aumentos de color/geometría/mosaico (ver `configs/config.yaml`, `train.py`).
- **Métricas propias** con `pycocotools`; los subgrupos de robustez se evalúan ignorando las cajas de otros grupos.
- **Oclusión:** CCT-20 no tiene etiquetas de oclusión, así que usamos un proxy por sustracción de fondo (`occlusion.py`); no está validado con etiquetas manuales.
- **Arquitectura:** usamos YOLOv8 de Ultralytics; lo implementado por nosotros es el pipeline de datos, la evaluación, Grad-CAM, el proxy de oclusión y el monitoreo.

## Reproducibilidad

Semilla 42 y `deterministic=True`. Las versiones usadas (`torch 2.11`, `ultralytics 8.4.193`, Python 3.13) están en `results/decisions.json`.
Nuestra corrida: GPU A100, 30 épocas, lote 32, 1.14 h. Los datos y los pesos (`*.pt`) no se suben a GitHub (están en `.gitignore`).

## Datos y referencias

Caltech Camera Traps, LILA BC. Beery, Van Horn y Perona, *Recognition in Terra Incognita*, ECCV 2018 (<https://openaccess.thecvf.com/content_ECCV_2018/html/Beery_Recognition_in_Terra_ECCV_2018_paper.html>).
La lista completa de referencias está en `report/refs.bib`.
