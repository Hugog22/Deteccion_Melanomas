# 🔬 Detección Temprana de Melanoma con Deep Learning y XAI

> **Pipeline de Deep Learning Clínicamente Informado para Dermoscopía v3.1**  
> Protocolo: TRIPOD+AI | Dataset: ISIC 2020 | Arquitectura: EfficientNet-B4

---

## 📐 Arquitectura del Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                  DATASET ISIC 2020 DERMATOSCÓPICO                │
│    33 126 imágenes | Benigno (~98.2%) | Maligno (~1.8%)          │
│    patient_id verificado vía Kaggle Ground Truth                 │
└──────────────────────────────┬──────────────────────────────────┘
                               │
        ┌──────────────────────▼───────────────────────┐
        │          FASE 1: PREPROCESAMIENTO             │
        │  • ISICArchiveDownloader / ISIC2020KaggleLoader│
        │  • patient_level_split() — split por paciente │
        │  • Circular crop (elimina viñeteado)          │
        │  • WeightedRandomSampler + Focal Loss (γ=2)  │
        │  • Augmentation clínico (CLAHE, Albumentations)│
        │  • CutMix (p=0.3) + MixUp (p=0.2)            │
        └──────────────────────┬───────────────────────┘
                               │
        ┌──────────────────────▼───────────────────────┐
        │      FASE 2: MODELADO DL + ABLACIONES        │
        │  • EfficientNet-B4 (pre-entrenado ImageNet)   │
        │  • Resolución nativa: 380×380                 │
        │  • Head: BN → Drop(0.5) → 512 → GELU → BN   │
        │    → Drop(0.3) → 1 (logit)                   │
        │  • AdamW + LR diferencial (backbone/head)     │
        │  • Warmup lineal + CosineAnnealing            │
        │  • EMA (decay=0.999) + Mixed Precision        │
        │  • BatchNorm backbone CONGELADO siempre       │
        │  • COND-A/B/C: análisis principal             │
        │  • ABLAC-D: sin WeightedSampler               │
        │  • ABLAC-E: sin Focal Loss                    │
        └──────────────────────┬───────────────────────┘
                               │
        ┌──────────────────────▼───────────────────────┐
        │          FASE 3: XAI — GRAD-CAM++             │
        │  • Hooks forward/backward en features[-1]     │
        │  • Supresión de artefactos de padding          │
        │  • Normalización por percentiles (p2–p98)     │
        │  • TTA debiasing (4 augmentaciones geom.)     │
        │  • AOS real por imagen (máscara binaria)      │
        │  • Detección Clever Hans: HoughLines + HSV    │
        └──────────────────────┬───────────────────────┘
                               │
        ┌──────────────────────▼───────────────────────┐
        │          FASE 4: EVALUACIÓN CLÍNICA v3.1      │
        │  • Umbral calibrado en VAL, frozen en TEST    │
        │  • Temperature Scaling (Platt, L-BFGS)        │
        │  • TTA en inferencia (10 augmentaciones)      │
        │  • AUC-ROC + Sens/Spec + IC BCa 95% (n=2000) │
        │  • DeLong test (A vs C, A vs B) — ΔAUC        │
        │  • Mann-Whitney U AOS + effect size r         │
        └──────────────────────────────────────────────┘
```

---

## 🗂️ Estructura del Proyecto

```
Detección_Melanomas/
├── 📓 melanoma_detection.ipynb       # Notebook: interfaz visual sobre run_pipeline.py
├── 🐍 run_pipeline.py                # Orquestador principal v3.1 (A/B/C + ablaciones D/E)
├── 🐍 evaluate_trained_model.py      # Evaluar checkpoint existente (Fases 3-6) sin re-entrenar
├── 🐍 run_xai_only.py                # Recalcular mapas Grad-CAM++ sin re-entrenar
├── 📋 requirements.txt               # Dependencias
├── src/
│   ├── data_pipeline.py              # Fase 1: Dataset + Augmentation + FocalLoss + CutMix/MixUp
│   ├── model.py                      # Fase 2: EfficientNet-B4 + EMA + Trainer
│   ├── xai_gradcam.py                # Fase 3: Grad-CAM++ + Clever Hans + TTA debiasing
│   ├── evaluation.py                 # Fase 4: Métricas clínicas + Calibración + Bootstrap
│   └── modify_pipeline.py            # Script de migración/refactoring automático
├── tests/                            # Pruebas unitarias y de regresión
│   ├── test_occlusion.py             # Test de oclusión de cuadrante (sesgo espacial)
│   ├── test_occlusion_controlled.py  # Test controlado de 4 cuadrantes (64×64)
│   └── test_regressions.py           # Regresiones: split, loaders, bias, calibración
├── data/                             # Dataset descargado o generado automáticamente
│   ├── raw/benign/ | raw/malignant/
│   └── metadata.csv                  # Índice maestro con labels, split y metadatos
├── checkpoints/
│   └── main/                         # best_model.pth + ema_shadow
└── results/
    ├── gradcam_*.png
    ├── evaluation_*.png
    └── final_results_v3.1.json       # Resumen estructurado del último run
```

> `data/`, `checkpoints/` y `results/` son artefactos generados.
> Pueden quedar desalineados si conservas salidas de ejecuciones antiguas.
> La fuente de verdad del pipeline es `run_pipeline.py` + `src/`.

---

## 🧠 Modelo: EfficientNet-B4

| Propiedad | Valor |
|-----------|-------|
| **Backbone** | `efficientnet_b4` (ImageNet pre-entrenado) |
| **Resolución de entrada** | 380 × 380 px |
| **Dim. de features** | 1792 |
| **Head de clasificación** | BN(1792) → Dropout(0.5) → Linear(512) → GELU → BN(512) → Dropout(0.3) → Linear(1) |
| **Inicialización** | Xavier Normal (pesos), cero (bias) |
| **Freeze strategy** | `partial`: bloques `features[:5]` congelados, `features[5:]` entrenables |
| **BatchNorm backbone** | Siempre congelado (`eval()` + `requires_grad=False`) |
| **Optimizador** | AdamW — LR backbone: 1e-4, LR head: 1e-3, weight decay: 1e-4 |
| **Scheduler** | LinearLR warmup (5 epochs, factor=0.1) → CosineAnnealingLR (eta_min=1e-7) |
| **EMA** | Decay 0.999, shadow weights usados en validación e inferencia |
| **Precisión** | Mixed precision (`torch.autocast`) |
| **Early stopping** | Patience 15, monitoreado en val AUC |
| **Gradient clipping** | max_norm = 1.0 |

---

## 🚀 Instalación y Ejecución

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Ejecutar pipeline v3.1 (recomendado)

```bash
python3 run_pipeline.py
```

### Experimentos incluidos

- `COND-A`: baseline sesgado, sin mitigación (sin `RandomResizedCrop`).
- `COND-B`: mitigación parcial con `RandomResizedCrop(scale=(0.65, 0.90))`.
- `COND-C`: mitigación real con `MaskedISICDataset` (máscaras de segmentación).
- `ABLAC-D`: igual que `COND-B`, pero sin `WeightedRandomSampler`.
- `ABLAC-E`: igual que `COND-B`, pero sin `Focal Loss`.

> El análisis clínico principal compara **A/B/C**.  
> **D/E** son ablaciones de sensibilidad para aislar el efecto del sampler y de la pérdida focal.

### 3. Evaluar un modelo ya entrenado

```bash
python3 evaluate_trained_model.py
```

Carga `best_model.pth` (incluyendo pesos EMA si existen), aplica Temperature Scaling en VAL, evalúa en TEST con TTA, y genera `final_results_eval_only.json`.

### 4. Recalcular XAI sin re-entrenar

```bash
python3 run_xai_only.py
```

Genera mapas Grad-CAM++ sobre muestras balanceadas del test set y guarda `gradcam_analysis_fixed.png`.

### 5. Ejecutar Notebook (interfaz visual)

```bash
jupyter notebook melanoma_detection.ipynb
```

### 6. Ejecutar tests

```bash
python3 -m pytest tests/
```

### 7. Protección contra leakage

- Por defecto, `run_pipeline.py` exige agrupamiento estrictamente válido por paciente.
- Si `metadata.csv` contiene `patient_id_source=isic_id_fallback`, el script aborta antes de entrenar para evitar claims científicos no defendibles.
- Solo desactiva esta protección si quieres una corrida exploratoria y vas a presentar los resultados como tales.

---

## 🧮 Fundamentos Matemáticos

### Focal Loss (Lin et al., 2017)

$$\text{FL}(p_t) = -\alpha_t \cdot (1 - p_t)^\gamma \cdot \log(p_t)$$

donde:
- $p_t$: probabilidad de la clase correcta
- $\alpha_t$: peso de la clase malignant = 0.75, benign = 0.25
- $\gamma = 2$: parámetro de enfoque (suprime ejemplos fáciles)

### Grad-CAM++ (Chattopadhay et al., 2018)

$$L^c_\text{GradCAM++} = \text{ReLU}\left(\sum_k \alpha^c_k \cdot A^k\right)$$

$$\alpha^c_k = \frac{\partial^2 y^c / \partial (A^k)^2}{2\partial^2 y^c / \partial (A^k)^2 + \sum_{ij} A^k_{ij} \cdot \partial^3 y^c / \partial (A^k)^3}$$

### Artifact Overlap Score (AOS)

$$\text{AOS} = \frac{\sum_{p \in \text{artifact}} \text{CAM}(p)}{\sum_{p} \text{CAM}(p)}$$

AOS alto → modelo enfocado en marcadores quirúrgicos o reglas en vez de la lesión (efecto Clever Hans).

### Umbral Clínico Óptimo

$$\theta^* = \arg\min_\theta \text{FPR}(\theta) \quad \text{s.t.} \quad \text{TPR}(\theta) \geq 0.90$$

### Temperature Scaling

$$p = \sigma\!\left(\frac{\text{logit}}{T}\right)$$

$T$ optimizado vía L-BFGS en el conjunto de validación para calibrar probabilidades.

---

## 📊 Métricas de Rendimiento (Contexto Oncológico)

| Métrica | Prioridad | Justificación Clínica |
|---------|-----------|----------------------|
| **Sensibilidad (Recall)** | 🔴 PRIMARIA | FN = melanoma no detectado → fatal |
| AUC-ROC | Alta | Discriminación global robusta al umbral |
| Especificidad | Media | FP = biopsia innecesaria (costo, ansiedad) |
| F1-Score | Media | Balance Sens/Prec para reporting |
| Accuracy | Baja | Engañosa con clases desbalanceadas |

> **Target clínico:** Sensibilidad ≥ 90% con Especificidad ≥ 70%

### Resultados Benchmark (ISIC 2020 completo — 33 126 imágenes)

| Métrica | Valor | IC 95% (BCa) |
|---------|-------|-------------|
| **AUC-ROC** | 0.8584 | 0.8148 – 0.8888 |
| **Sensibilidad** | 0.9062 | 0.8312 – 0.9529 |
| **Especificidad** | 0.6380 | 0.6236 – 0.6517 |
| Umbral calibrado | 0.5441 | — |
| Mejor AUC validación | 0.8849 | — |
| Temperature (T) | 1.2160 | — |

---

## 🔧 Técnicas Avanzadas

### Preprocesamiento clínico
- **Circular crop**: elimina viñeteado del dermatoscopio (presente en 15.9% maligno vs 2.7% benigno).
- **CenterCrop 85%**: elimina transiciones de borde externo.
- **CutMix/MixUp**: regularización con etiquetas continuas durante entrenamiento.

### XAI robusto
- **Supresión de artefactos de padding**: elimina activaciones espurias en bordes causadas por padding asimétrico de EfficientNet.
- **Normalización por percentiles (p2–p98)**: evita que outliers de un solo píxel compriman activaciones legítimas.
- **TTA debiasing**: promedio de heatmaps sobre 4 transformaciones geométricas con inversión espacial.
- **Detector clásico de artefactos**: `HoughLinesP` para reglas de medición + HSV thresholding para marcas de violeta genciana.

### Evaluación estadística
- **Bootstrap BCa**: 2000 remuestreos con jackknife para intervalos de confianza.
- **DeLong test**: comparación no paramétrica de curvas AUC-ROC entre condiciones.
- **Mann-Whitney U**: test de sesgo AOS con corrección Bonferroni y effect size $r$.

---

## ⚠️ Limitaciones y Uso Responsable

> **AVISO:** Este pipeline es de investigación académica. No debe usarse para diagnóstico clínico real sin:
> - Validación externa en cohortes independientes
> - Calibración de probabilidades validada prospectivamente
> - Evaluación de equidad por fototipo de piel (Fitzpatrick scale)
> - Aprobación regulatoria (FDA 510(k) / CE Mark en la UE)
> - Validación prospectiva en entorno clínico controlado

### Limitaciones actuales del repo

- El split solo puede llamarse "por paciente" si `patient_id_source` no usa `isic_id_fallback`.
- `final_results_v3.1.json` representa únicamente el último run ejecutado con el código actual; no debe mezclarse con checkpoints o figuras heredadas de corridas antiguas.
- La prevalencia del estudio reportada en los resultados se calcula desde `metadata.csv`, no desde los valores nominales de `CONFIG`.
- La especificidad actual (63.8%) está por debajo del target clínico de 70%, lo que implica una tasa elevada de falsos positivos.

---

*Desarrollado siguiendo el protocolo TRIPOD+AI para transparencia y reproducibilidad en modelos de IA médica.*
