# 🔬 Detección Temprana de Melanoma con Deep Learning y XAI

> **Pipeline de Deep Learning Clínicamente Informado para Dermoscopía**  
> Protocolo: TRIPOD+AI | Dataset: ISIC | Arquitectura: DenseNet-121

---

## 📐 Arquitectura del Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATASET ISIC DERMATOSCÓPICO                   │
│          Clase 0: Benigno (~80%) | Clase 1: Maligno (~20%)       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
        ┌──────────────────────▼───────────────────────┐
        │          FASE 1: PREPROCESAMIENTO             │
        │  • ISICArchiveDownloader (API real) +         │
        │    ISICSyntheticUnitTest (solo CI/CD)         │
        │  • patient_level_split() — split por paciente │
        │  • WeightedRandomSampler + Focal Loss (γ=2)  │
        │  • Augmentation clínico (CLAHE, flips, color) │
        └──────────────────────┬───────────────────────┘
                               │
        ┌──────────────────────▼───────────────────────┐
        │      FASE 2: MODELADO DL + ABLACIONES        │
        │  • DenseNet-121 (pre-entrenado ImageNet)      │
        │  • Fine-tuning: denseblock4 + norm5           │
        │    (bloques 1-3 + transition3 CONGELADOS)     │
        │  • COND-A/B/C: análisis principal             │
        │  • ABLAC-D: sin WeightedSampler               │
        │  • ABLAC-E: sin Focal Loss                    │
        └──────────────────────┬───────────────────────┘
                               │
        ┌──────────────────────▼───────────────────────┐
        │          FASE 3: XAI — GRAD-CAM++             │
        │  • Hooks forward/backward en denseblock4      │
        │  • AOS real por imagen (máscara binaria)      │
        │  • Detección visual del sesgo Clever Hans     │
        └──────────────────────┬───────────────────────┘
                               │
        ┌──────────────────────▼───────────────────────┐
        │          FASE 4: EVALUACIÓN CLÍNICA v2.0      │
        │  • Umbral calibrado en VAL, frozen en TEST    │
        │  • AUC-ROC + Sens/Spec + IC BCa 95%          │
        │  • DeLong test (A vs C, A vs B) — ΔAUC        │
        │  • Mann-Whitney U AOS + effect size r         │
        │  • Prevalencia recalibrada (Bayes, ~1.8%)     │
        └──────────────────────────────────────────────┘
```

---

## 🗂️ Estructura del Proyecto

```
Detección_Melanomas/
├── 📓 melanoma_detection.ipynb    # Notebook: interfaz sobre run_pipeline.py v2.0
├── 🐍 run_pipeline.py             # Orquestador principal v2.0 (A/B/C + ablaciones D/E)
├── 📋 requirements.txt            # Dependencias
├── src/
│   ├── data_pipeline.py           # Fase 1: Dataset + Augmentation + FocalLoss
│   ├── model.py                   # Fase 2: DenseNet-121 + Trainer
│   ├── xai_gradcam.py             # Fase 3: Grad-CAM++ + Clever Hans
│   └── evaluation.py              # Fase 4: Métricas clínicas + Mitigación
├── data/                          # Dataset descargado o generado automáticamente
│   ├── raw/benign/ | raw/malignant/
│   └── metadata.csv               # Índice maestro con labels, split y metadatos
├── checkpoints/
│   ├── unmitigated/ | partial/ | masked/
│   └── ablation_no_sampler/ | ablation_no_focal/
└── results/
    ├── dataset_samples.png
    ├── gradcam_cond_a.png
    ├── evaluation_unmitigated.png
    ├── evaluation_partial.png
    ├── evaluation_masked.png
    └── final_results_v2.json      # Resumen estructurado del último run
```

> `data/`, `checkpoints/` y `results/` son artefactos generados.
> Pueden quedar desalineados si conservas salidas de ejecuciones antiguas.
> La fuente de verdad del pipeline es `run_pipeline.py` + `src/`.

---

## 🚀 Instalación y Ejecución

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Ejecutar pipeline v2.0 (recomendado)

```bash
python3 run_pipeline.py
```

### Experimentos incluidos

- `COND-A`: baseline sesgado, sin mitigación.
- `COND-B`: mitigación parcial con `RandomResizedCrop` en entrenamiento.
- `COND-C`: mitigación real con `MaskedISICDataset`.
- `ABLAC-D`: igual que `COND-B`, pero sin `WeightedSampler`.
- `ABLAC-E`: igual que `COND-B`, pero sin `Focal Loss`.

> El análisis clínico principal compara **A/B/C**.  
> **D/E** son ablaciones de sensibilidad para aislar el efecto del sampler y de la pérdida focal.

### 3. Ejecutar Notebook (interfaz visual)

```bash
jupyter notebook melanoma_detection.ipynb
```

### 4. Protección contra leakage

- Por defecto, `run_pipeline.py` exige agrupamiento estrictamente válido por paciente.
- Si `metadata.csv` contiene `patient_id_source=isic_id_fallback`, el script aborta antes de entrenar para evitar claims científicos no defendibles.
- Solo desactiva esta protección si quieres una corrida exploratoria y vas a presentar los resultados como tales.

---

## 🧮 Fundamentos Matemáticos

### Focal Loss (Lin et al., 2017)

$$\text{FL}(p_t) = -\alpha_t \cdot (1 - p_t)^\gamma \cdot \log(p_t)$$

donde:
- $p_t$: probabilidad de la clase correcta
- $\alpha_t$: peso por clase ($\alpha_\text{maligno} \approx 0.80$, $\alpha_\text{benigno} \approx 0.20$)
- $\gamma = 2$: parámetro de enfoque (suprime ejemplos fáciles)

### Grad-CAM++ (Chattopadhay et al., 2018)

$$L^c_\text{GradCAM++} = \text{ReLU}\left(\sum_k \alpha^c_k \cdot A^k\right)$$

$$\alpha^c_k = \frac{\partial^2 y^c / \partial (A^k)^2}{2\partial^2 y^c / \partial (A^k)^2 + \sum_{ij} A^k_{ij} \cdot \partial^3 y^c / \partial (A^k)^3}$$

### Umbral Clínico Óptimo

$$\theta^* = \arg\min_\theta \text{FPR}(\theta) \quad \text{s.t.} \quad \text{TPR}(\theta) \geq 0.90$$

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

---

## ⚠️ Limitaciones y Uso Responsable

> **AVISO:** Este pipeline es de investigación académica. No debe usarse para diagnóstico clínico real sin:
> - Validación externa en cohortes independientes
> - Calibración de probabilidades (Platt scaling)  
> - Evaluación de equidad por fototipo de piel (Fitzpatrick scale)
> - Aprobación regulatoria (FDA 510(k) / CE Mark en la UE)
> - Validación prospectiva en entorno clínico controlado

### Limitaciones actuales del repo

- El split solo puede llamarse "por paciente" si `patient_id_source` no usa `isic_id_fallback`.
- `final_results_v2.json` representa únicamente el último run ejecutado con el código actual; no debe mezclarse con checkpoints o figuras heredadas de corridas antiguas.
- La prevalencia del estudio reportada en los resultados se calcula desde `metadata.csv`, no desde los valores nominales de `CONFIG`.

---

*Desarrollado siguiendo el protocolo TRIPOD+AI para transparencia y reproducibilidad en modelos de IA médica.*
