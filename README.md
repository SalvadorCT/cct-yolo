# Monitoreo de fauna con camaras trampa (YOLO) - Caltech Camera Traps (CCT-20)

Pregunta 3 del parcial 2026-II (Redes Neuronales y Aprendizaje Profundo, UNI).
Pipeline reproducible: descarga de LILA BC -> dataset YOLO sin fuga entre secuencias -> fine-tuning de YOLOv8 ->
evaluacion por especie -> generalizacion a camaras nuevas (+ Grad-CAM) -> robustez (oclusion / tamano) -> reporte de actividad.

## Donde correr cada cosa
| Que | Donde |
|---|---|
| Todo lo pesado (descarga ~6 GB, entrenamiento, evaluacion) | **Google Colab con GPU T4** -> `notebooks/Colab_Pipeline.ipynb` (alternativa: Kaggle con GPU, mismos comandos) |
| Prueba rapida del codigo (sin GPU ni descargas) | tu PC: `python tests/smoke_test.py` y `python tests/test_eval.py` |
| Informe | Overleaf: sube la carpeta `report/` |

Desde la raiz del repo, los modulos se llaman con `PYTHONPATH=src python -m cct.<modulo>` (el notebook ya lo configura).

## Pasos (Colab)
1. Sube el repo a GitHub (ver abajo) o `cct_yolo.zip` a tu Drive.
2. Abre `notebooks/Colab_Pipeline.ipynb` en Colab, activa GPU y ejecuta las celdas en orden.
3. Los resultados quedan en `MyDrive/cct_yolo_out/` (pesos, tablas CSV/LaTeX, figuras).
4. Copia tablas y figuras a `report/` y reemplaza los `\TODO{}` del informe con tus numeros.

## Entregable -> archivo
| # | Entregable | Codigo | Salida (en `reports/`) |
|---|---|---|---|
| 1 | YOLO fine-tuned y decisiones (resolucion, aumentos, vacias como fondo) | `train.py`, `build_dataset.py`, `configs/config.yaml` | `runs/<nombre>/decisions.json` |
| 2 | mAP@0.5 y @0.5:0.95 por especie, efecto del desbalance | `evaluate.py` | `tables/summary.csv`, `per_class_*.csv/.tex`, `common_vs_rare_*.csv`, `figures/ap_vs_train_instances_*.png` |
| 3 | Generalizacion eccv_train / eccv_val + Grad-CAM | `generalization.py`, `gradcam.py` | `tables/generalization_*.csv`, `figures/gradcam_seen_vs_unseen.png` |
| 4 | Robustez por oclusion y tamano | `occlusion.py`, `robustness.py` | `tables/robustness_*.csv`, `figures/robustness_groups.png` |
| 5 | Modulo de monitoreo ecologico | `monitoring.py` | `monitoring/loc<id>_{activity,summary}_*.csv`, `_report.md`, `.png` |
| 6 | Discusion de IA responsable | - | `report/main.tex` |

## Decisiones de diseno (resumen)
- **Particiones**: `eccv_val` (ubicaciones nuevas) = `test_unseen`. De `eccv_train` se separa por *secuencia* (hash de `seq_id`) val y `test_seen`, para que fotos casi identicas de una rafaga no caigan en train y test.
- **Imagenes vacias** como fondo (archivo de etiquetas vacio): submuestreadas en train (`train_empty_ratio`), completas en evaluacion para medir falsos positivos.
- **Desbalance**: repeat-factor sampling (Gupta et al., LVIS) para especies raras, aplicado duplicando lineas de `train.txt`.
- **Resolucion 640** (las fotos vienen a 1024 px; animales pequenos se pierden con resoluciones menores). Aumentos `strong` vs `baseline` para la ablacion.
- **Metricas** con pycocotools propio (`eval_utils.py`), AP por clase desde la matriz de precision. Los subgrupos de robustez se evaluan *ignorando* las GT de otros grupos (area fuera de rango).
- **Oclusion**: CCT20 no trae etiquetas. Se usa un proxy por sustraccion de fondo por camara (ver `occlusion.py`), validable con etiquetas manuales (Spearman). Es una aproximacion, y el informe lo dice.
- **Grad-CAM** propio sobre la capa 9 (SPPF) del backbone; mide cuanta energia cae dentro de la caja y si el modelo sigue detectando al tapar al animal.

## Reproducibilidad
Semilla 42, `deterministic=True`; versiones en `runs/<nombre>/pip_freeze.txt`. Todos los parametros estan en `configs/config.yaml` y se pueden cambiar con `--set clave=valor` (p. ej. `--set train.epochs=5`).

## Subir a GitHub
```bash
cd cct_yolo
git init && git add . && git commit -m "Pregunta 3: pipeline CCT20 + YOLOv8"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/cct-yolo.git
git push -u origin main
```
`data/`, `runs/`, `reports/` y `*.pt` estan en `.gitignore` (no se suben).

## Datos y licencia
Caltech Camera Traps, LILA BC (CDLA-Permissive-1.0). Cita: Beery, Van Horn, Perona, *Recognition in Terra Incognita*, ECCV 2018.
