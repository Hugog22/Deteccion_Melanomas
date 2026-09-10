"""
================================================================================
EVALUACIÓN DE MODELO YA ENTRENADO (sin re-entrenar)
================================================================================
Script: evaluate_trained_model.py

Por qué existe este script:
  El entrenamiento (Fase 2 de run_pipeline.py) ya se completó en el cluster y
  el checkpoint "best_model.pth" ya está guardado en disco. Este script
  reproduce exactamente las Fases 3-6 de run_pipeline.py (calibración,
  evaluación clínica, XAI y guardado de resultados) PERO cargando los pesos
  del checkpoint en vez de llamar a trainer.train().

Requisitos para que esto funcione:
  1) CONFIG["data_dir"] debe apuntar EXACTAMENTE al mismo directorio que se
     usó durante el entrenamiento, y "metadata.csv" debe existir ahí con la
     columna "split" ya calculada (el patient_level_split original). Si se
     recalcula el split aquí con otra semilla o de otra forma, el test set
     podría no ser el mismo que "vio" el modelo como held-out → resultados
     inválidos.
  2) CONFIG["checkpoint_dir"] debe apuntar al directorio que contiene
     "best_model.pth" (el guardado por ModelTrainer._save_checkpoint()).
  3) img_size / freeze_strategy deben coincidir con los usados en el
     entrenamiento (solo afectan la arquitectura/transforms, no los pesos).

Qué hace, paso a paso:
  FASE 3: Temperature Scaling sobre VAL (igual que run_pipeline.py)
  FASE 4: Evaluación clínica con TTA + umbral calibrado en VAL, frozen en TEST
  FASE 5: XAI Grad-CAM++ (opcional, no crítico si falla o falta xai_gradcam.py)
  FASE 6: Guardado de resultados finales (JSON) e impresión de tabla resumen
================================================================================
"""

import os
import sys
import json
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src"))
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — debe coincidir con la usada durante el entrenamiento
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Rutas — DEBEN apuntar al MISMO sitio que usó run_pipeline.py al entrenar
    "data_dir":         "./data",
    "checkpoint_dir":    "./checkpoints/main",
    "results_dir":       "./results",

    # Arquitectura / datos — deben coincidir con el entrenamiento
    "img_size":          224,
    "freeze_strategy":   "partial",   # solo afecta requires_grad, no la arquitectura
    "batch_size":         64,
    "num_workers":        8,

    # EMA — debe coincidir con lo usado en entrenamiento (para saber si aplicar ema_shadow)
    "use_ema":            True,
    "ema_decay":          0.999,

    # Inferencia
    "use_tta":            True,
    "n_tta_augmentations": 6,
    "use_temperature_scaling": True,

    # Clínico
    "sensitivity_target": 0.90,
    "threshold_objective": "sensitivity",

    # Estadística
    "n_bootstrap":        2000,

    # XAI
    "n_xai_samples":      6,

    "seed": 42,
}


def main():
    print("=" * 75)
    print("   EVALUACIÓN DE MODELO YA ENTRENADO (sin re-entrenar)")
    print("=" * 75)

    os.makedirs(CONFIG["results_dir"], exist_ok=True)

    checkpoint_path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"❌ No se encontró el checkpoint en '{checkpoint_path}'.\n"
            "   Revisa que CONFIG['checkpoint_dir'] apunte al directorio correcto "
            "del cluster (el que contenía 'best_model.pth')."
        )

    metadata_path = Path(CONFIG["data_dir"]) / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"❌ No se encontró '{metadata_path}'.\n"
            "   Este archivo (con la columna 'split' ya calculada) es el que se "
            "generó durante el entrenamiento. Sin él no se puede reconstruir el "
            "mismo val/test set que vio el modelo — cópialo desde el cluster."
        )

    # ── Dispositivo ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n🖥️  GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\n🖥️  Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("\n🖥️  CPU (la inferencia con TTA puede ir lenta)")

    import pandas as pd
    from data_pipeline import set_global_seed, ISICDataset
    from torch.utils.data import DataLoader

    set_global_seed(CONFIG["seed"])

    # ═══════════════════════════════════════════════════════════════════════
    # DATOS: reconstruir SOLO val/test (no hace falta el train set para evaluar)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 75)
    print("  CARGANDO METADATA Y RECONSTRUYENDO VAL/TEST")
    print("═" * 75)

    df = pd.read_csv(metadata_path)
    if "split" not in df.columns:
        raise RuntimeError(
            "❌ 'metadata.csv' no tiene columna 'split'. No se puede evaluar sin "
            "el split original — no recalcules patient_level_split aquí, usa el "
            "metadata.csv exacto que generó el entrenamiento."
        )

    datasets = {
        split: ISICDataset(df, split=split, img_size=CONFIG["img_size"],
                            condition="partial", return_path=False)
        for split in ["val", "test"]
    }
    loaders = {
        split: DataLoader(
            datasets[split], batch_size=CONFIG["batch_size"], shuffle=False,
            num_workers=CONFIG["num_workers"], pin_memory=True
        )
        for split in ["val", "test"]
    }

    n_total = len(df)
    n_benign = int((df["label"] == 0).sum())
    n_malignant = int((df["label"] == 1).sum())

    # ═══════════════════════════════════════════════════════════════════════
    # MODELO: reconstruir arquitectura y cargar pesos del checkpoint
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 75)
    print("  CARGANDO CHECKPOINT — SIN RE-ENTRENAR")
    print("═" * 75)

    from model import MelanomaClassifier, EMA

    model = MelanomaClassifier(freeze_strategy=CONFIG["freeze_strategy"])
    model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Réplica exacta de lo que hace ModelTrainer.train() al final: si se usó
    # EMA, los pesos "buenos" son los EMA (shadow), no los crudos guardados
    # en model_state_dict — hay que aplicarlos igual que en el entrenamiento
    # original para evaluar el mismo modelo que se hubiese evaluado allí.
    if CONFIG["use_ema"] and "ema_shadow" in checkpoint:
        ema = EMA(model, decay=CONFIG["ema_decay"])
        ema.shadow = checkpoint["ema_shadow"]
        ema.apply_shadow(model)
        print("   ✅ Pesos EMA (shadow) aplicados — igual que al final del entrenamiento original.")
    else:
        print("   ⚠️  Sin EMA en el checkpoint — usando pesos crudos.")

    model.eval()

    history = checkpoint.get("history")
    best_val_auc = checkpoint.get("val_auc")
    trained_epoch = checkpoint.get("epoch")
    print(f"   ✅ Checkpoint cargado (época {trained_epoch}, Val AUC={best_val_auc:.4f})")

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 3: CALIBRACIÓN DE PROBABILIDADES — TEMPERATURE SCALING
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 75)
    print("  FASE 3: CALIBRACIÓN — TEMPERATURE SCALING")
    print("═" * 75)

    from evaluation import (
        ClinicalEvaluator, TestTimeAugmentation, TemperatureScaling
    )

    if CONFIG["use_temperature_scaling"]:
        temp_scaler = TemperatureScaling()
        temp_scaler.fit(model, loaders["val"], device)
    else:
        temp_scaler = None

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 4: EVALUACIÓN CON TTA + UMBRAL FROZEN
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 75)
    print("  FASE 4: EVALUACIÓN — TTA + UMBRAL CALIBRADO EN VAL")
    print("═" * 75)

    evaluator = ClinicalEvaluator(
        model=model, device=device,
        clinical_sensitivity_target=CONFIG["sensitivity_target"],
        threshold_objective=CONFIG["threshold_objective"],
    )

    if CONFIG["use_tta"]:
        print("\n📈 Calibrando umbral con TTA...")
        val_labels, val_probs = TestTimeAugmentation.predict_with_tta(
            model, loaders["val"], device, CONFIG["n_tta_augmentations"]
        )
        evaluator._val_y_true = val_labels
        evaluator._val_y_prob = val_probs

        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(val_labels, val_probs)
        valid_mask = (tpr >= CONFIG["sensitivity_target"])
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) > 0:
            best_idx = valid_idx[np.argmin(fpr[valid_idx])]
            threshold = float(thresholds[best_idx])
        else:
            youden_idx = np.argmax(tpr - fpr)
            threshold = float(thresholds[youden_idx])
        evaluator._calibrated_threshold = threshold
        print(f"   θ* (TTA, VAL) = {threshold:.4f}")

        print("\n📊 Evaluando TEST con TTA...")
        test_labels, test_probs = TestTimeAugmentation.predict_with_tta(
            model, loaders["test"], device, CONFIG["n_tta_augmentations"]
        )
        evaluator._test_y_true = test_labels
        evaluator._test_y_prob = test_probs
        metrics = evaluator._compute_metrics(test_labels, test_probs, threshold)
    else:
        threshold = evaluator.calibrate_threshold(loaders["val"])
        metrics = evaluator.evaluate_test(loaders["test"], threshold=threshold)

    cis = evaluator.compute_bootstrap_cis(metrics, n_bootstrap=CONFIG["n_bootstrap"])

    fig = evaluator.plot_comprehensive_evaluation(
        metrics=metrics, cis=cis, history=history,
        save_path=os.path.join(CONFIG["results_dir"], "evaluation_main.png")
    )
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 5: XAI — GRAD-CAM++ (opcional, no crítico)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 75)
    print("  FASE 5: XAI — GRAD-CAM++")
    print("═" * 75)

    try:
        from xai_gradcam import CleverHansAnalyzer
        from data_pipeline import get_transforms, apply_circular_crop

        target_layer = model.get_features_layer()
        analyzer = CleverHansAnalyzer(
            model=model, target_layer=target_layer,
            device=device, use_gradcam_pp=True
        )

        test_df = df[df["split"] == "test"].copy()
        xai_samples = []
        transform_test = get_transforms("test", CONFIG["img_size"], condition="partial")

        n_per_class = CONFIG["n_xai_samples"] // 2
        test_benign = test_df[test_df["label"] == 0].sample(
            n=min(n_per_class, len(test_df[test_df["label"] == 0])),
            random_state=CONFIG["seed"]
        )
        test_malignant = test_df[test_df["label"] == 1].sample(
            n=min(n_per_class, len(test_df[test_df["label"] == 1])),
            random_state=CONFIG["seed"]
        )
        xai_selection = pd.concat([test_benign, test_malignant])

        for _, row in xai_selection.iterrows():
            try:
                img = Image.open(row["image_path"]).convert("RGB")
                img_np = np.array(img)
                img_np = apply_circular_crop(img_np)
                img_tensor = transform_test(image=img_np)["image"]
                artifact_type = row.get("artifact_type", "none")
                xai_samples.append((img_tensor, row["label"], artifact_type, row["image_path"]))
            except Exception:
                pass

        if xai_samples:
            analyzer.visualize_clever_hans_effect(
                xai_samples,
                save_path=os.path.join(CONFIG["results_dir"], "gradcam_analysis.png"),
                figsize=(22, 5)
            )
            plt.close()
            print("✅ Análisis Grad-CAM++ completado.")
        else:
            print("⚠️  No hay muestras suficientes para Grad-CAM.")
    except Exception as e:
        print(f"⚠️  Grad-CAM++ falló (no crítico): {e}")

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 6: RESULTADOS FINALES
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 75)
    print("  RESULTADOS FINALES")
    print("═" * 75)

    key_metrics = ["AUC-ROC", "AUC-PR", "Sensitivity", "Specificity",
                   "Precision", "F1-Score", "Accuracy", "FN", "FP"]

    print(f"\n{'Métrica':<20} {'Valor':>10} {'IC 95%':>25}")
    print("─" * 60)
    for m in key_metrics:
        val = metrics[m]
        ci_key = {"AUC-ROC": "auc_roc", "Sensitivity": "sensitivity",
                  "Specificity": "specificity", "F1-Score": "f1",
                  "Precision": "ppv"}.get(m)
        if ci_key and ci_key in cis:
            _, lo, hi = cis[ci_key]
            print(f"{m:<20} {val:>10.4f} [{lo:.4f} – {hi:.4f}]")
        else:
            print(f"{m:<20} {val:>10}")

    fn = metrics["FN"]
    fp = metrics["FP"]
    n_test_mal = int((df[df["split"] == "test"]["label"] == 1).sum())
    n_test_ben = int((df[df["split"] == "test"]["label"] == 0).sum())
    print(f"\n🏥 Impacto clínico:")
    print(f"   Melanomas en test: {n_test_mal}")
    print(f"   Melanomas detectados: {n_test_mal - fn} ({(n_test_mal - fn)/max(n_test_mal,1):.1%})")
    print(f"   Melanomas NO detectados (FN): {fn}")
    print(f"   Falsos positivos: {fp} de {n_test_ben} benignos ({fp/max(n_test_ben,1):.1%})")

    def _safe(obj):
        if isinstance(obj, (np.integer, np.floating)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, torch.device): return str(obj)
        return str(obj)

    results = {
        "config": {k: v for k, v in CONFIG.items()},
        "checkpoint_source": checkpoint_path,
        "trained_epoch": trained_epoch,
        "dataset": {
            "total": n_total, "benign": n_benign, "malignant": n_malignant,
            "prevalence": n_malignant / n_total
        },
        "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else int(v)
                    for k, v in metrics.items() if not k.startswith("_")},
        "confidence_intervals": {
            k: {"estimate": float(est), "ci_lower": float(lo), "ci_upper": float(hi)}
            for k, (est, lo, hi) in cis.items()
        },
        "temperature": temp_scaler.temperature if temp_scaler else 1.0,
        "best_val_auc": float(best_val_auc) if best_val_auc is not None else None,
        "threshold": float(threshold),
    }

    results_path = os.path.join(CONFIG["results_dir"], "final_results_eval_only.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=_safe)

    print(f"\n✅ Evaluación completada sin re-entrenar.")
    print(f"📁 Resultados en: {CONFIG['results_dir']}/")


if __name__ == "__main__":
    main()
