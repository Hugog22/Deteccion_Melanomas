import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_pipeline import (
    create_dataloaders,
    patient_level_split,
    summarize_grouping_quality,
    validate_grouping_quality,
)
from evaluation import ClinicalEvaluator
from run_pipeline import summarize_loaded_dataset
from xai_gradcam import CleverHansAnalyzer


class DummyModel:
    def eval(self):
        return self


class RegressionTests(unittest.TestCase):
    def test_summarize_grouping_quality_detects_isic_fallback(self):
        df = pd.DataFrame(
            {
                "patient_id": ["p1", "ISIC_1", "ISIC_2"],
                "patient_id_source": [
                    "clinical_or_unstructured",
                    "isic_id_fallback",
                    "isic_id_fallback",
                ],
                "label": [0, 1, 0],
            }
        )

        summary = summarize_grouping_quality(df)

        self.assertEqual(summary["fallback_count"], 2)
        self.assertAlmostEqual(summary["fallback_fraction"], 2 / 3)
        self.assertFalse(summary["strict_patient_grouping"])
        self.assertEqual(summary["grouping_level"], "mixed_patient_image")

    def test_validate_grouping_quality_raises_when_strict_mode_is_required(self):
        grouping_summary = {
            "fallback_count": 4,
            "fallback_fraction": 0.5,
            "strict_patient_grouping": False,
        }

        with self.assertRaises(RuntimeError):
            validate_grouping_quality(grouping_summary, require_strict_patient_grouping=True)

    def test_summarize_loaded_dataset_uses_actual_label_counts(self):
        df = pd.DataFrame(
            {
                "label": [0, 0, 0, 1],
                "split": ["train", "train", "test", "test"],
            }
        )

        summary = summarize_loaded_dataset(df)

        self.assertEqual(summary["n_benign"], 3)
        self.assertEqual(summary["n_malignant"], 1)
        self.assertAlmostEqual(summary["study_prevalence"], 0.25)
        self.assertEqual(summary["split_counts"]["test"], 2)

    def test_patient_level_split_keeps_all_splits_non_empty_when_possible(self):
        df = pd.DataFrame(
            {
                "patient_id": [f"b{i}" for i in range(3)] + [f"m{i}" for i in range(3)],
                "label": [0, 0, 0, 1, 1, 1],
            }
        )

        split_df = patient_level_split(df, seed=7)

        split_counts = split_df["split"].value_counts().to_dict()
        self.assertEqual(split_counts.get("train"), 2)
        self.assertEqual(split_counts.get("val"), 2)
        self.assertEqual(split_counts.get("test"), 2)

    def test_create_dataloaders_keeps_small_train_loader_non_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            rows = []
            for idx, (split, label) in enumerate(
                [
                    ("train", 0),
                    ("train", 1),
                    ("train", 0),
                    ("val", 1),
                    ("test", 0),
                ]
            ):
                image_path = tmp_path / f"img_{idx}.jpg"
                image = np.full((256, 256, 3), 32 * (idx + 1), dtype=np.uint8)
                Image.fromarray(image).save(image_path)
                rows.append(
                    {
                        "image_path": str(image_path),
                        "label": label,
                        "split": split,
                    }
                )

            df = pd.DataFrame(rows)
            loaders, datasets = create_dataloaders(
                df,
                img_size=224,
                batch_size=16,
                num_workers=0,
                condition="partial",
                use_weighted_sampler=True,
            )

            self.assertEqual(len(datasets["train"]), 3)
            self.assertEqual(len(loaders["train"]), 1)

    def test_batch_bias_analysis_uses_peripheral_metric_in_classical_mode(self):
        analyzer = CleverHansAnalyzer.__new__(CleverHansAnalyzer)
        analyzer.model = DummyModel()
        analyzer.device = torch.device("cpu")
        analyzer.gradcam = lambda tensor, target_class=1: (
            np.ones((8, 8), dtype=np.float32),
            1,
            0.75,
        )
        analyzer.detect_artifacts_classical = (
            lambda image_path: (image_path.endswith("1.jpg"), "ruler")
        )

        loader = [
            (
                torch.zeros(2, 3, 8, 8),
                torch.tensor([0, 1]),
                ["img_0.jpg", "img_1.jpg"],
            )
        ]
        df_metadata = pd.DataFrame(
            {
                "image_path": ["img_0.jpg", "img_1.jpg"],
                "has_artifact": [False, False],
            }
        )

        summary = analyzer.batch_bias_analysis(loader, df_metadata=df_metadata, max_samples=2)

        self.assertEqual(summary["_detection_mode"], "classical_cv")
        self.assertEqual(summary["_score_kind"], "peripheral_act")
        self.assertEqual(len(summary["_mw_with"]), 1)
        self.assertEqual(len(summary["_mw_without"]), 1)
        self.assertEqual(summary["_aos_with"], [])

    def test_clinical_evaluator_precision_mode_maximizes_recall_once_ppv_target_is_met(self):
        evaluator = ClinicalEvaluator(
            model=DummyModel(),
            device=torch.device("cpu"),
            threshold_objective="precision",
            clinical_precision_target=0.75,
            minimum_recall_for_precision=0.40,
        )
        evaluator._run_inference_internal = lambda loader: (
            np.array([1, 1, 1, 0, 0, 0]),
            np.array([0.95, 0.85, 0.60, 0.80, 0.40, 0.20]),
        )

        threshold = evaluator.calibrate_threshold(val_loader=None)

        self.assertAlmostEqual(threshold, 0.60, places=6)

    def test_clinical_evaluator_precision_mode_falls_back_to_best_available_ppv(self):
        evaluator = ClinicalEvaluator(
            model=DummyModel(),
            device=torch.device("cpu"),
            threshold_objective="precision",
            clinical_precision_target=0.80,
            minimum_recall_for_precision=0.30,
        )
        evaluator._run_inference_internal = lambda loader: (
            np.array([1, 1, 1, 0, 0, 0]),
            np.array([0.90, 0.70, 0.60, 0.95, 0.80, 0.50]),
        )

        threshold = evaluator.calibrate_threshold(val_loader=None)

        self.assertAlmostEqual(threshold, 0.60, places=6)


if __name__ == "__main__":
    unittest.main()
