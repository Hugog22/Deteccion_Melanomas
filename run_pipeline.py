"""
================================================================================
PIPELINE MAESTRO v3.0 — FOCO EN PRECISIÓN MÁXIMA
================================================================================
Script: run_pipeline.py — v3.0

CAMBIOS RESPECTO A v2.0:
  - Arquitectura: EfficientNet-B4 (resolución 380×380)
  - Cabeza: FC(1792→512→1) con GELU, BatchNorm, Dropout
  - Entrenamiento: Warmup + CosineAnnealing, Gradient Accumulation, EMA
  - Augmentación: Agresiva (GridDistortion, CutMix, MixUp)
  - Inferencia: Test-Time Augmentation (TTA, 6 augmentaciones)
  - Calibración: Temperature Scaling sobre validación
  - Pérdida: Focal Loss binaria (salida única, BCEWithLogitsLoss)
  - Simplificación: 1 condición principal (partial) + ablaciones opcionales
  - Dataset: Descarga optimizada de ISIC con filtro por diagnóstico

================================================================================
"""

import os
import sys
import json
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # Backend no-interactivo para servidores
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent / "src"))
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DEL EXPERIMENTO
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Dataset
    "data_dir":        "./data",
    "n_benign":        None,
    "n_malignant":     None,
    # 224→380: EfficientNet-B4 (torchvision, IMAGENET1K_V1) está pre-entrenada
    # nativamente a 380×380 — a 224 estábamos desalineados con esa resolución
    # y además usando el dataset local de Kaggle, que NO son thumbnails de
    # 224px (esos venían solo del fallback de descarga por API); las JPEG
    # reales de ISIC 2020 tienen resolución bastante mayor, así que 380 sigue
    # siendo un downscale, no un upscale. Sube el uso de VRAM/cómputo por
    # imagen, lo que ayuda a mantener la GPU ocupada (ver nota en batch_size).
    "img_size":        380,

    # Entrenamiento
    # 64→96: con la resolución subida a 380 el forward/backward ya pesa más
    # por imagen; L40S (48GB) sigue teniendo margen de sobra con AMP. Un
    # batch más grande manda más imágenes de una vez a la GPU (lo que
    # comentabas de "aumentar las fotos que se envían a la gráfica"), reduce
    # el overhead relativo de cada lanzamiento de kernel/CPU→GPU y ayuda a
    # que quede menos tiempo la GPU esperando. Vigila `nvidia-smi` en las
    # primeras épocas: si sobra VRAM de verdad, se puede subir más (p.ej.
    # 128); si ves OOM, bájalo.
    "batch_size":      96,
    "num_epochs":      60,
    "lr_backbone":     1e-4,
    "lr_head":         1e-3,
    "weight_decay":    1e-4,
    "warmup_epochs":   5,
    "focal_gamma":     2.0,
    "focal_alpha":     0.75,   # Peso para clase positiva (maligno)
    "freeze_strategy": "partial",
    "patience":        15,
    # Experimento: solo entrenar la cabeza (profundidad 0), backbone
    # congelado el resto del entrenamiento — sin más descongelaciones.
    "max_unfreeze_depth":   0,
    "unfreeze_start_epoch": 1,
    "unfreeze_interval":    8,

    # Experimento condición C (mitigación real): la lesión se recorta por
    # segmentación DURANTE el entrenamiento (MaskedISICDataset), no solo en
    # inferencia — el modelo nunca ve rotuladores/burbujas/artefactos fuera
    # de la lesión. Aplica a train/val/test por igual.
    "condition":  "masked",
    # ISIC 2020 no trae máscaras de segmentación oficiales para sus propias
    # imágenes (esas son de los challenges 2016/2017/2018 Task 1, con
    # overlap de IDs incierto). En vez de aceptar el fallback por Otsu,
    # `lesion_segmentation.py` entrena un U-Net con los pares reales
    # imagen/máscara de ISIC 2018 Task 1 (oficiales, ~2694 pares) y lo usa
    # para PREDECIR una máscara para cada una de las 33,126 imágenes de
    # ISIC 2020 antes de arrancar el entrenamiento — así masks_dir llega
    # siempre poblado y _load_mask nunca cae al Otsu. Sigue siendo una
    # predicción de modelo, no una anotación humana verificada para esas
    # 33,126 imágenes en concreto (esas no existen) — ver el aviso de
    # honestidad al principio de lesion_segmentation.py.
    "masks_dir":  "./data/isic2020_masks",
    "seg_data_dir": "./data/isic2018_task1",
    "seg_checkpoint": "./checkpoints/lesion_unet.pt",
    "seg_epochs": 20,
    "gradient_accumulation_steps": 1,  # Batch efectivo = batch_size (96 * 1)
    "use_ema":         True,
    "ema_decay":       0.999,
    "label_smoothing": 0.1,


    # Inferencia
    "use_tta":         True,
    # 6→10: más augmentaciones de test-time reducen la varianza de la
    # probabilidad estimada por imagen, lo que da un umbral de decisión más
    # estable en VAL y métricas de TEST más fiables. Es prácticamente gratis
    # con la GPU libre que tienes (TTA es solo inferencia, no backprop).
    "n_tta_augmentations": 10,
    "use_temperature_scaling": True,

    # Clínico
    "sensitivity_target": 0.90,
    "threshold_objective": "sensitivity",

    # Estadística
    "n_bootstrap":     2000,

    # XAI
    "n_xai_samples":   6,     # 3 benignos + 3 malignos (selección balanceada)

    # Salidas
    "checkpoint_dir":   "./checkpoints/main",
    "results_dir":      "./results",
    "seed":             42,
    # Metadatos oficiales del ISIC 2020 Challenge (Kaggle: "SIIM-ISIC Melanoma
    # Classification", archivo ISIC_2020_Training_GroundTruth_v2.csv). Si existe
    # en esta ruta, se usa para sustituir el patient_id por el REAL antes de
    # hacer el split. Sin este archivo, la API pública de ISIC no expone
    # patient_id real y el split degenera en split por imagen (fuga de datos).
    "isic2020_ground_truth_csv": "./data/ISIC_2020_Training_GroundTruth_v2.csv",
    # Carpeta con las imágenes JPEG del dataset ISIC 2020 (Kaggle), descargadas
    # y descomprimidas localmente. Si esta carpeta Y el CSV de arriba existen,
    # se usan como fuente PRIORITARIA en vez de la API pública de ISIC Archive:
    # patient_id viene garantizado al 100% real, sin fallback ni enriquecimiento.
    "isic2020_images_dir": "./data/isic2020_jpeg",
    # ANTES estaba en False y, además, el flag no se leía en ningún sitio del
    # código (validate_grouping_quality() nunca se llamaba): daba una falsa
    # sensación de seguridad. Ahora SÍ se aplica en patient_level_split(), y
    # por defecto está en True: si no se consigue agrupamiento real por
    # paciente, el pipeline se detiene en vez de reportar métricas infladas.
    "require_strict_patient_grouping": True,

    # Dataset ISIC — modo demostración
    "demo_mode":        False
}


def main():
    print("=" * 75)
    print("   PIPELINE DE DETECCIÓN DE MELANOMA v3.1")
    print("   EfficientNet-B4 + Circular Crop + Focal Loss + TTA + Grad-CAM++")
    print("=" * 75)

    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # ── Dispositivo ────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n🖥️  GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("\n🖥️  Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        print("\n🖥️  CPU (entrenamiento lento)")

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 1: DATOS
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 75)
    print("  FASE 1: ADQUISICIÓN DE DATOS")
    print("═" * 75)

    from data_pipeline import (
        set_global_seed, ISICArchiveDownloader, ISICSyntheticUnitTest,
        ISIC2020KaggleLoader, patient_level_split,
        enrich_with_isic2020_patient_ids, create_dataloaders, FocalLoss
    )
    import pandas as pd

    set_global_seed(CONFIG["seed"])

    metadata_path = Path(CONFIG["data_dir"]) / "metadata.csv"
    using_real_data = False
    data_source_label = "SINTÉTICO [⚠️ SOLO DEMO]"

    def _grouped_split_or_fail(df_in: pd.DataFrame) -> pd.DataFrame:
        """
        Envoltorio de patient_level_split() que, si el agrupamiento por
        paciente no es defendible (require_strict_patient_grouping=True y
        seguimos en fallback por isic_id), falla con un mensaje accionable
        en vez de dejar que el RuntimeError crudo de validate_grouping_quality
        llegue sin contexto.
        """
        try:
            return patient_level_split(
                df_in, patient_col="patient_id",
                require_strict_patient_grouping=CONFIG["require_strict_patient_grouping"]
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"{e}\n\n"
                "➡️  Opciones:\n"
                f"   1) Descarga 'ISIC_2020_Training_GroundTruth_v2.csv' (Kaggle, "
                "competición 'SIIM-ISIC Melanoma Classification') y colócalo en "
                f"'{CONFIG['isic2020_ground_truth_csv']}' para tener patient_id real.\n"
                "   2) Si solo quieres una corrida exploratoria (NO válida para "
                "publicar ni para reportar como definitiva), pon "
                "CONFIG['require_strict_patient_grouping'] = False de forma "
                "explícita y documenta esa limitación en cualquier resultado."
            ) from e

    kaggle_loader = ISIC2020KaggleLoader(
        images_dir=CONFIG["isic2020_images_dir"],
        ground_truth_csv=CONFIG["isic2020_ground_truth_csv"],
    )

    if metadata_path.exists():
        print("📂 Cargando dataset existente...")
        df = pd.read_csv(metadata_path)
        using_real_data = "isic_id" in df.columns
        # Solo re-enriquecemos si el cacheado no vino ya 100% garantizado
        # desde el dataset local de Kaggle (ese caso no tiene nada que enriquecer).
        already_guaranteed = using_real_data and (df["patient_id_source"] == "isic_2020_ground_truth").all()
        if using_real_data and not already_guaranteed:
            df = enrich_with_isic2020_patient_ids(
                df, CONFIG["isic2020_ground_truth_csv"]
            )
        data_source_label = "REAL (ISIC 2020 Kaggle, local)" if already_guaranteed else \
            ("REAL (ISIC Archive API)" if using_real_data else "SINTÉTICO [⚠️ SOLO DEMO]")
        if "split" not in df.columns and "patient_id" in df.columns:
            print("🧩 Recalculando split por paciente...")
            df = _grouped_split_or_fail(df)
            df.to_csv(metadata_path, index=False)

    elif kaggle_loader.available():
        # Fuente PRIORITARIA: patient_id 100% real, sin fallback ni enriquecimiento.
        print("📦 Dataset ISIC 2020 (Kaggle) encontrado localmente. Usándolo como fuente.")
        df = kaggle_loader.load(
            n_benign=CONFIG["n_benign"],
            n_malignant=CONFIG["n_malignant"],
            seed=CONFIG["seed"],
        )
        using_real_data = True
        data_source_label = "REAL (ISIC 2020 Kaggle, local)"
        df = _grouped_split_or_fail(df)
        df.to_csv(metadata_path, index=False)

    else:
        print("🌐 Dataset local de Kaggle no encontrado. Intentando descarga vía API ISIC...")
        downloader = ISICArchiveDownloader(CONFIG["data_dir"])
        df = downloader.download_balanced_subset(
            n_benign=CONFIG["n_benign"],
            n_malignant=CONFIG["n_malignant"],
            img_size=CONFIG["img_size"],
            add_artifacts_for_demo=CONFIG["demo_mode"]
        )
        if df is not None:
            using_real_data = True
            data_source_label = "REAL (ISIC Archive API)"
            df = enrich_with_isic2020_patient_ids(
                df, CONFIG["isic2020_ground_truth_csv"]
            )
            df = _grouped_split_or_fail(df)
            df.to_csv(metadata_path, index=False)
            print("✅ Datos reales ISIC descargados y guardados.")
        else:
            print("\n⚠️  API ISIC no disponible. Activando generador sintético...")
            gen = ISICSyntheticUnitTest()
            df = gen.generate(
                CONFIG["data_dir"],
                n_benign=CONFIG["n_benign"],
                n_malignant=CONFIG["n_malignant"],
                img_size=CONFIG["img_size"]
            )
            # Datos sintéticos: patient_id ya es sintético 1:1 por imagen, no
            # hay isic_id real que enriquecer ni fuga que prevenir de verdad
            # (esto es solo para probar que el pipeline corre de punta a punta).
            df = patient_level_split(
                df, patient_col="patient_id",
                require_strict_patient_grouping=False
            )
            df.to_csv(metadata_path, index=False)
            data_source_label = "SINTÉTICO [⚠️ SOLO DEMO]"

    # Estadísticas del dataset
    n_total = len(df)
    n_benign = int((df["label"] == 0).sum())
    n_malignant = int((df["label"] == 1).sum())
    data_type = data_source_label
    print(f"\n📊 Tipo de datos: {data_type}")
    print(f"   Total: {n_total} | Benigno: {n_benign} | Maligno: {n_malignant}")
    print(f"   Prevalencia: {n_malignant/n_total:.1%}")

    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        print(f"   {split}: {len(sub)} ({(sub['label']==1).sum()} malignos)")

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 2: ENTRENAMIENTO — MODELO PRINCIPAL
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 75)
    print("  FASE 2: ENTRENAMIENTO — EfficientNet-B4 + FOCAL LOSS")
    print("═" * 75)

    from model import MelanomaClassifier, build_optimizer_and_scheduler, ModelTrainer

    # Condición 'masked': asegurar que TODAS las imágenes tienen una máscara
    # real (predicha por un segmentador entrenado con datos reales de ISIC
    # 2018 Task 1) antes de construir los dataloaders — así MaskedISICDataset
    # nunca cae al fallback por Otsu. Ver lesion_segmentation.py para el
    # detalle y el aviso de honestidad sobre qué es (predicción de modelo)
    # y qué no es (anotación humana verificada para estas 33,126 imágenes).
    if CONFIG["condition"] == "masked":
        from lesion_segmentation import ensure_real_masks_for_condition_masked
        ensure_real_masks_for_condition_masked(
            df,
            masks_dir=CONFIG["masks_dir"],
            seg_data_dir=CONFIG["seg_data_dir"],
            checkpoint_path=CONFIG["seg_checkpoint"],
            device=device,
            epochs=CONFIG["seg_epochs"],
        )

    # Crear dataloaders con augmentaciones agresivas y CutMix/MixUp
    loaders, datasets = create_dataloaders(
        df, img_size=CONFIG["img_size"],
        batch_size=CONFIG["batch_size"],
        num_workers=16,
        condition=CONFIG["condition"],
        mask_dir=CONFIG["masks_dir"],
        use_weighted_sampler=True,
        use_cutmix_mixup=True,
        preload_to_ram=True   # ← precarga en RAM (nodo tiene 503GB, dataset cabe de sobra)
    )

    # Verificación final: con las máscaras ya generadas arriba, esto debería
    # dar 0% en fallback Otsu (si no, algo falló en la generación — revisar
    # los avisos impresos por ensure_real_masks_for_condition_masked).
    if CONFIG["condition"] == "masked":
        from pathlib import Path as _Path
        _mask_dir = _Path(CONFIG["masks_dir"])
        for split in ["train", "val", "test"]:
            sub = df[df["split"] == split]
            n_total = len(sub)
            n_present = sum(
                (_mask_dir / f"{_Path(p).stem}_segmentation.png").exists()
                for p in sub["image_path"]
            )
            fallback_rate = (n_total - n_present) / max(n_total, 1)
            print(f"   [{split.upper()}] Máscara presente: {n_present}/{n_total} "
                  f"| {fallback_rate:.1%} caería en fallback Otsu")

    # Focal Loss binaria
    criterion = FocalLoss(
        alpha=CONFIG["focal_alpha"],
        gamma=CONFIG["focal_gamma"]
    )

    # Modelo (pesos frescos ImageNet)
    model = MelanomaClassifier(freeze_strategy=CONFIG["freeze_strategy"])

    # Optimizador con warmup + cosine annealing
    optimizer, scheduler = build_optimizer_and_scheduler(
        model,
        lr_backbone=CONFIG["lr_backbone"],
        lr_head=CONFIG["lr_head"],
        weight_decay=CONFIG["weight_decay"],
        n_epochs=CONFIG["num_epochs"],
        warmup_epochs=CONFIG["warmup_epochs"]
    )

    # Trainer con EMA, gradient accumulation, progressive unfreezing
    trainer = ModelTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=CONFIG["checkpoint_dir"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        use_ema=CONFIG["use_ema"],
        ema_decay=CONFIG["ema_decay"],
        label_smoothing=CONFIG["label_smoothing"],
        patience=CONFIG["patience"]
    )

    history = trainer.train(
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        num_epochs=CONFIG["num_epochs"],
        max_unfreeze_depth=CONFIG["max_unfreeze_depth"],
        unfreeze_start_epoch=CONFIG["unfreeze_start_epoch"],
        unfreeze_interval=CONFIG["unfreeze_interval"]
    )

    # Guardar historial
    with open(os.path.join(CONFIG["checkpoint_dir"], "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 3: CALIBRACIÓN DE PROBABILIDADES
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 75)
    print("  FASE 3: CALIBRACIÓN — TEMPERATURE SCALING")
    print("═" * 75)

    from evaluation import (
        ClinicalEvaluator, StatisticalValidator,
        TestTimeAugmentation, TemperatureScaling
    )

    # Temperature Scaling (sobre validación)
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

    # Calibrar umbral en VAL (con TTA si habilitado)
    if CONFIG["use_tta"]:
        print("\n📈 Calibrando umbral con TTA...")
        val_labels, val_probs = TestTimeAugmentation.predict_with_tta(
            model, loaders["val"], device, CONFIG["n_tta_augmentations"]
        )
        # Guardar internamente para calibración del umbral
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

        # Evaluar en TEST con TTA
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

    # IC BCa 95%
    cis = evaluator.compute_bootstrap_cis(
        metrics, n_bootstrap=CONFIG["n_bootstrap"]
    )

    # Figura de evaluación
    fig = evaluator.plot_comprehensive_evaluation(
        metrics=metrics, cis=cis, history=history,
        save_path=os.path.join(CONFIG["results_dir"], "evaluation_main.png")
    )
    plt.close(fig)

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 5: XAI — GRAD-CAM++
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 75)
    print("  FASE 5: XAI — GRAD-CAM++")
    print("═" * 75)

    try:
        from xai_gradcam import CleverHansAnalyzer

        target_layer = model.get_features_layer()
        analyzer = CleverHansAnalyzer(
            model=model, target_layer=target_layer,
            device=device, use_gradcam_pp=True
        )

        test_df = df[df["split"] == "test"].copy()
        xai_samples = []
        # Usamos datasets["test"] (ya construido con CONFIG["condition"]) en
        # vez de reconstruir el tensor a mano con condition="partial" fijo.
        # Antes esto hacía que el panel de Grad-CAM++ se generase SIEMPRE
        # sobre imágenes sin enmascarar, aunque el modelo se hubiera
        # entrenado en condición 'masked' — comparación inválida. Así queda
        # garantizado el mismo preprocesado que vio el modelo, sea cual sea
        # la condición activa.
        test_dataset = datasets["test"]
        path_to_idx = {p: i for i, p in enumerate(test_dataset.df["image_path"])}

        # Selección BALANCEADA: 3 benignos + 3 malignos (no solo los primeros 6)
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
                idx = path_to_idx.get(row["image_path"])
                if idx is None:
                    continue
                img_tensor, _ = test_dataset[idx]
                artifact_type = row.get("artifact_type", "none")
                xai_samples.append((img_tensor, row["label"], artifact_type, row["image_path"]))
            except Exception:
                pass

        if xai_samples:
            fig_xai = analyzer.visualize_clever_hans_effect(
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

    # Impacto clínico
    fn = metrics["FN"]
    fp = metrics["FP"]
    n_test_mal = int((df[df["split"] == "test"]["label"] == 1).sum())
    n_test_ben = int((df[df["split"] == "test"]["label"] == 0).sum())
    print(f"\n🏥 Impacto clínico:")
    print(f"   Melanomas en test: {n_test_mal}")
    print(f"   Melanomas detectados: {n_test_mal - fn} ({(n_test_mal - fn)/max(n_test_mal,1):.1%})")
    print(f"   Melanomas NO detectados (FN): {fn}")
    print(f"   Falsos positivos: {fp} de {n_test_ben} benignos ({fp/max(n_test_ben,1):.1%})")

    # Guardar resultados
    def _safe(obj):
        if isinstance(obj, (np.integer, np.floating)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, torch.device): return str(obj)
        return str(obj)

    results = {
        "config": {k: v for k, v in CONFIG.items()},
        "using_real_data": using_real_data,
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
        "best_val_auc": float(trainer.best_auc),
        "threshold": float(threshold),
    }

    results_path = os.path.join(CONFIG["results_dir"], "final_results_v3.1.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=_safe)

    print(f"\n✅ Pipeline v3.0 completado.")
    print(f"📁 Resultados en: {CONFIG['results_dir']}/")
    if not using_real_data:
        print("\n⚠️  RECORDATORIO: Resultados sobre datos SINTÉTICOS.")
        print("   Para análisis válido, usar datos reales ISIC.")


if __name__ == "__main__":
    import pandas as pd
    main()
