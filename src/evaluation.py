"""
================================================================================
FASE 4 (REVISADA): EVALUACIÓN CLÍNICA CON RIGOR ESTADÍSTICO
================================================================================
Módulo: evaluation.py — v2.0

CORRECCIONES APLICADAS respecto a v1.0:

  [FIX-CRÍTICO] Fuga de datos en umbral de decisión:
      v1.0: threshold calculado sobre el TEST set → métricas optimistamente sesgadas.
      v2.0: ClinicalEvaluator.calibrate_threshold() DEBE llamarse sobre el VAL set.
            evaluate_test() recibe el threshold como argumento frozen. Si no se
            llama calibrate_threshold() primero, raise RuntimeError.

  [NEW-1] StatisticalValidator:
      - bootstrap_ci(): Intervalos de confianza BCa del 95% para todas las métricas.
        n_bootstrap=2000 por defecto (suficiente para CI de AUC en n>50 muestras).
      - delong_test(): Test de DeLong (1988) para comparar dos AUC-ROC pareados.
        Implementación via método de componentes estructurales.
      - mann_whitney_bias(): Mann-Whitney U para comparar AOS con/sin artefacto.
        Incluye effect size r = Z / sqrt(N).

  [FIX-2] Umbral arbitrario del sesgo (peripheral > center * 0.8):
      Reemplazado por test estadístico formal (Mann-Whitney U) con
      corrección de Bonferroni si hay comparaciones múltiples.

  [FIX-3] Comparación base vs. mitigado:
      Reemplaza "mostrar dos números" por DeLong test (z, p-value, 95% CI del ΔAUC).

Referencia estadística:
    DeLong ER et al. "Comparing the areas under two or more correlated receiver
    operating characteristic curves: a nonparametric approach."
    Biometrics 1988;44(3):837–845. DOI:10.2307/2531595

    Efron B, Tibshirani R. "An Introduction to the Bootstrap." Chapman & Hall, 1993.
    (Capítulo 14: Bootstrap confidence intervals BCa)
================================================================================
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score, accuracy_score, recall_score, precision_score
)
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 4.0: TESTS ESTADÍSTICOS FORMALES
# ─────────────────────────────────────────────────────────────────────────────
class StatisticalValidator:
    """
    Validación estadística rigurosa de los resultados del modelo.

    Implementa los tres tests necesarios para un análisis científicamente sólido:

    1. bootstrap_ci():
       Intervalos de confianza del 95% por método BCa (Bias-Corrected and
       Accelerated bootstrap). Recomendado sobre el percentile bootstrap simple
       cuando la distribución del estimador es asimétrica (frecuente en AUC).

    2. delong_test():
       Test de DeLong (1988) para comparar dos AUC-ROC pareados (mismo test set).
       El método de componentes estructurales es exacto, no paramétrico y más
       potente que un bootstrap no pareado.

    3. mann_whitney_bias():
       Mann-Whitney U para comparar el AOS (Artifact Overlap Score) entre
       imágenes con y sin artefactos. Incluye effect size r = Z/√N.
    """

    # ── 4.0.1: Bootstrap BCa CI ───────────────────────────────────────────
    @staticmethod
    def bootstrap_ci(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        metric: str = "auc_roc",
        threshold: float = 0.5,
        n_bootstrap: int = 2000,
        confidence: float = 0.95,
        seed: int = 42
    ) -> Tuple[float, float, float]:
        """
        Calcula el intervalo de confianza BCa para la métrica especificada.

        BCa bootstrap es preferido sobre percentile bootstrap cuando la
        distribución del estimador puede ser asimétrica (común para AUC
        cuando la muestra es pequeña o el desbalance es alto).

        Args:
            y_true:      Etiquetas verdaderas (0/1).
            y_prob:      Probabilidades de la clase positiva.
            metric:      'auc_roc', 'sensitivity', 'specificity', 'f1', 'ppv'.
            threshold:   Umbral de clasificación (para métricas binarias).
            n_bootstrap: Número de remuestras bootstrap (≥2000 para CI estables).
            confidence:  Nivel de confianza (0.95 → IC 95%).
            seed:        Semilla para reproducibilidad.

        Returns:
            (estimado_original, ci_lower, ci_upper)
        """
        rng = np.random.RandomState(seed)

        def _compute_metric(yt, yp, thresh):
            if metric == "auc_roc":
                try: return roc_auc_score(yt, yp)
                except: return 0.5
            yb = (yp >= thresh).astype(int)
            if metric == "sensitivity": return recall_score(yt, yb, pos_label=1, zero_division=0)
            if metric == "specificity": return recall_score(yt, yb, pos_label=0, zero_division=0)
            if metric == "f1":          return f1_score(yt, yb, pos_label=1, zero_division=0)
            if metric == "ppv":
                from sklearn.metrics import precision_score
                return precision_score(yt, yb, pos_label=1, zero_division=0)
            raise ValueError(f"Métrica '{metric}' no reconocida.")

        original_estimate = _compute_metric(y_true, y_prob, threshold)
        n = len(y_true)

        # ── Bootstrap distribution ─────────────────────────────────────────
        boot_estimates = []
        for _ in range(n_bootstrap):
            idx = rng.randint(0, n, n)
            boot_yt = y_true[idx]
            boot_yp = y_prob[idx]
            # Solo calcular si ambas clases presentes (necesario para AUC)
            if len(np.unique(boot_yt)) < 2:
                continue
            boot_estimates.append(_compute_metric(boot_yt, boot_yp, threshold))

        boot_estimates = np.array(boot_estimates)

        if len(boot_estimates) < 100:
            warnings.warn(
                f"Solo {len(boot_estimates)} remuestras válidas. "
                "El IC puede ser inestable con n pequeña."
            )
            alpha = (1 - confidence) / 2
            return (original_estimate,
                    float(np.percentile(boot_estimates, 100 * alpha)),
                    float(np.percentile(boot_estimates, 100 * (1 - alpha))))

        # ── BCa correction ─────────────────────────────────────────────────
        # Sesgo: proporción de bootstraps inferiores a la estimación original
        prop = np.mean(boot_estimates < original_estimate)
        prop = np.clip(prop, 1e-6, 1 - 1e-6)  # Prevenir -inf o inf en ppf
        z0 = scipy_stats.norm.ppf(prop)

        # Aceleración: jackknife influence function
        jack_estimates = []
        for i in range(n):
            yt_j = np.delete(y_true, i)
            yp_j = np.delete(y_prob, i)
            if len(np.unique(yt_j)) < 2:
                jack_estimates.append(original_estimate)
                continue
            jack_estimates.append(_compute_metric(yt_j, yp_j, threshold))
        jack_estimates = np.array(jack_estimates)
        jack_mean = jack_estimates.mean()
        jack_diff = jack_mean - jack_estimates
        acc = np.sum(jack_diff ** 3) / (6.0 * (np.sum(jack_diff ** 2) ** 1.5 + 1e-12))

        alpha = (1 - confidence) / 2
        z_alpha_lo = scipy_stats.norm.ppf(alpha)
        z_alpha_hi = scipy_stats.norm.ppf(1 - alpha)

        # Percentiles BCa
        alpha_lo = scipy_stats.norm.cdf(
            z0 + (z0 + z_alpha_lo) / (1 - acc * (z0 + z_alpha_lo))
        )
        alpha_hi = scipy_stats.norm.cdf(
            z0 + (z0 + z_alpha_hi) / (1 - acc * (z0 + z_alpha_hi))
        )

        alpha_lo = np.clip(alpha_lo, 0.001, 0.499)
        alpha_hi = np.clip(alpha_hi, 0.501, 0.999)

        ci_lo = float(np.percentile(boot_estimates, 100 * alpha_lo))
        ci_hi = float(np.percentile(boot_estimates, 100 * alpha_hi))

        return original_estimate, ci_lo, ci_hi

    # ── 4.0.2: DeLong Test ────────────────────────────────────────────────
    @staticmethod
    def delong_test(
        y_true: np.ndarray,
        y_prob_a: np.ndarray,
        y_prob_b: np.ndarray
    ) -> Dict:
        """
        Test de DeLong (1988) para comparar dos AUC-ROC sobre el mismo test set.

        Fundamento matemático:
            El test usa el método de componentes estructurales (Mann-Whitney
            formulation) para calcular la varianza de la diferencia de AUC sin
            hacer suposiciones paramétricas, aprovechando que los scores son
            pareados (mismo paciente evaluado por ambos modelos).

            Para n muestras positivas (m) y negativas (n):
            AUC_A = Σ_i Σ_j I(score_A_i > score_A_j) / (m*n)

            La varianza de ΔAUC = AUC_A - AUC_B tiene en cuenta la correlación
            entre los dos modelos mediante los componentes estructurales:
            V_A[i] = (1/m) * Σ_j I(score_A_j > score_A_i)  [para positivos]
            W_A[j] = (1/n) * Σ_i I(score_A_i > score_A_j)  [para negativos]

            Var(AUC) = (Var(V) + Var(W)) / (m * n)

        Args:
            y_true:   Etiquetas verdaderas (0/1).
            y_prob_a: Probabilidades del Modelo A (baseline).
            y_prob_b: Probabilidades del Modelo B (mitigado).

        Returns:
            Dict con: z_stat, p_value, delta_auc, auc_a, auc_b,
                      ci_95_lo, ci_95_hi, interpretation.
        """
        pos_mask = (y_true == 1)
        neg_mask = (y_true == 0)

        scores_pos_a = y_prob_a[pos_mask]
        scores_neg_a = y_prob_a[neg_mask]
        scores_pos_b = y_prob_b[pos_mask]
        scores_neg_b = y_prob_b[neg_mask]

        m = len(scores_pos_a)   # Positivos
        n = len(scores_neg_a)   # Negativos

        if m < 2 or n < 2:
            warnings.warn(
                "DeLong test requiere ≥2 muestras positivas y negativas. "
                "Usando bootstrap como fallback."
            )
            return StatisticalValidator._bootstrap_auc_comparison(
                y_true, y_prob_a, y_prob_b
            )

        def structural_components(scores_pos, scores_neg):
            """Calcula V (componente de positivos) y W (componente de negativos)."""
            V = np.array([
                np.mean(scores_neg < sp) + 0.5 * np.mean(scores_neg == sp)
                for sp in scores_pos
            ])
            W = np.array([
                np.mean(scores_pos > sn) + 0.5 * np.mean(scores_pos == sn)
                for sn in scores_neg
            ])
            return V, W

        # Componentes estructurales para ambos modelos
        V_a, W_a = structural_components(scores_pos_a, scores_neg_a)
        V_b, W_b = structural_components(scores_pos_b, scores_neg_b)

        # AUC como media de los componentes
        auc_a = V_a.mean()
        auc_b = V_b.mean()

        # Varianza y covarianza de la diferencia
        S_aa = (np.var(V_a, ddof=1) / m) + (np.var(W_a, ddof=1) / n)
        S_bb = (np.var(V_b, ddof=1) / m) + (np.var(W_b, ddof=1) / n)
        S_ab = (np.cov(V_a, V_b)[0, 1] / m) + (np.cov(W_a, W_b)[0, 1] / n)

        var_diff = S_aa + S_bb - 2 * S_ab
        se_diff  = np.sqrt(max(var_diff, 1e-12))

        delta_auc = auc_b - auc_a
        z_stat    = delta_auc / se_diff
        p_value   = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))  # Bilateral

        # IC 95% de ΔAUC
        ci_lo = delta_auc - 1.96 * se_diff
        ci_hi = delta_auc + 1.96 * se_diff

        interpretation = (
            f"ΔAUC = {delta_auc:+.4f} [{ci_lo:+.4f}, {ci_hi:+.4f}], "
            f"z = {z_stat:.3f}, p = {p_value:.4f}"
        )
        significance = "SIGNIFICATIVO (p<0.05)" if p_value < 0.05 else "No significativo"

        result = {
            "auc_a":        float(auc_a),
            "auc_b":        float(auc_b),
            "delta_auc":    float(delta_auc),
            "z_stat":       float(z_stat),
            "p_value":      float(p_value),
            "ci_95_lo":     float(ci_lo),
            "ci_95_hi":     float(ci_hi),
            "se_diff":      float(se_diff),
            "interpretation": interpretation,
            "significance": significance
        }

        print(f"\n📊 Test de DeLong (AUC-ROC A vs. B):")
        print(f"   AUC Modelo A (base):       {auc_a:.4f}")
        print(f"   AUC Modelo B (mitigado):   {auc_b:.4f}")
        print(f"   ΔAUC = B - A:              {delta_auc:+.4f}")
        print(f"   95% IC de ΔAUC:            [{ci_lo:+.4f}, {ci_hi:+.4f}]")
        print(f"   z = {z_stat:.3f}  |  p = {p_value:.4f}  ({significance})")

        return result

    @staticmethod
    def _bootstrap_auc_comparison(y_true, y_prob_a, y_prob_b,
                                   n_bootstrap=2000, seed=42) -> Dict:
        """Fallback bootstrap para DeLong cuando la muestra es insuficiente."""
        rng = np.random.RandomState(seed)
        auc_a = roc_auc_score(y_true, y_prob_a)
        auc_b = roc_auc_score(y_true, y_prob_b)
        delta = auc_b - auc_a
        boot_deltas = []
        for _ in range(n_bootstrap):
            idx = rng.randint(0, len(y_true), len(y_true))
            yt = y_true[idx]
            if len(np.unique(yt)) < 2: continue
            boot_deltas.append(
                roc_auc_score(yt, y_prob_b[idx]) - roc_auc_score(yt, y_prob_a[idx])
            )
        boot_deltas = np.array(boot_deltas)
        p = np.mean(np.abs(boot_deltas) >= abs(delta)) if len(boot_deltas) > 0 else 1.0
        return {
            "auc_a": auc_a, "auc_b": auc_b, "delta_auc": delta,
            "z_stat": np.nan, "p_value": float(p),
            "ci_95_lo": float(np.percentile(boot_deltas, 2.5)) if len(boot_deltas) > 0 else np.nan,
            "ci_95_hi": float(np.percentile(boot_deltas, 97.5)) if len(boot_deltas) > 0 else np.nan,
            "se_diff": np.nan, "interpretation": "Bootstrap (fallback)",
            "significance": "SIGNIFICATIVO (p<0.05)" if p < 0.05 else "No significativo"
        }

    # ── 4.0.3: Mann-Whitney U para sesgo ──────────────────────────────────
    @staticmethod
    def mann_whitney_bias(
        aos_with_artifact: List[float],
        aos_without_artifact: List[float],
        alpha_bonferroni: float = 1.0  # Ajustar si hay comparaciones múltiples
    ) -> Dict:
        """
        Test de Mann-Whitney U para comparar el Artifact Overlap Score (AOS)
        entre imágenes con y sin artefactos clínicos.

        H₀: La distribución de AOS es igual en ambos grupos.
        H₁: Las imágenes con artefactos tienen AOS significativamente mayor.
             (El modelo activa más en regiones periféricas cuando hay artefactos)

        El AOS es una métrica ordinal (rango 0-1), lo que hace que Mann-Whitney
        sea más apropiado que un t-test (no asume normalidad).

        Effect size r = Z / √N, donde:
            r < 0.1: despreciable
            r ∈ [0.1, 0.3): pequeño
            r ∈ [0.3, 0.5): mediano
            r ≥ 0.5: grande

        Args:
            aos_with_artifact:    Lista de AOS de imágenes con artefactos.
            aos_without_artifact: Lista de AOS de imágenes sin artefactos.
            alpha_bonferroni:     Factor de corrección Bonferroni (1.0 = sin corrección).

        Returns:
            Dict con: u_stat, p_value, effect_size_r, interpretation.
        """
        a = np.array(aos_with_artifact)
        b = np.array(aos_without_artifact)

        if len(a) < 3 or len(b) < 3:
            warnings.warn(
                f"Mann-Whitney requiere n≥3 por grupo. "
                f"Grupos: n_art={len(a)}, n_noart={len(b)}. "
                "Aumentar max_samples en batch_bias_analysis()."
            )
            return {"u_stat": np.nan, "p_value": 1.0, "effect_size_r": 0.0,
                    "n_art": len(a), "n_noart": len(b),
                    "mean_art": float(np.mean(a)) if len(a)>0 else np.nan,
                    "mean_noart": float(np.mean(b)) if len(b)>0 else np.nan,
                    "interpretation": "Muestra insuficiente"}

        n1, n2 = len(a), len(b)

        # Forzar método asintótico para que z sea coherente con U.
        # El método exacto (por defecto con n pequeña en scipy) no tiene
        # una fórmula z cerrada coherente con U, y reconstruirla via
        # norm.ppf(1 - p) sería circular. Con method="asymptotic" la
        # apro ximación normal es válida y z se calcula directamente:
        #   z = (U - μ_U) / σ_U   donde μ_U = n1*n2/2,  σ_U = sqrt(n1*n2*(n1+n2+1)/12)
        u_stat, p_value_raw = scipy_stats.mannwhitneyu(
            a, b, alternative="greater", method="asymptotic"
        )
        p_value = min(p_value_raw * alpha_bonferroni, 1.0)

        # z directo desde la estadística U (fórmula asintótica, sin ties)
        mu_u  = n1 * n2 / 2.0
        sig_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
        z = (u_stat - mu_u) / sig_u  # positivo → grupo 'a' mayor
        effect_size_r = abs(z) / np.sqrt(n1 + n2)

        magnitude = ("despreciable" if effect_size_r < 0.1 else
                     "pequeño"      if effect_size_r < 0.3 else
                     "mediano"      if effect_size_r < 0.5 else "grande")

        significance = ("SIGNIFICATIVO (p<0.05)" if p_value < 0.05
                        else "No significativo")

        result = {
            "u_stat":       float(u_stat),
            "p_value":      float(p_value),
            "p_value_raw":  float(p_value_raw),
            "effect_size_r": float(effect_size_r),
            "magnitude":    magnitude,
            "n_art":        len(a),
            "n_noart":      len(b),
            "mean_art":     float(np.mean(a)),
            "median_art":   float(np.median(a)),
            "mean_noart":   float(np.mean(b)),
            "median_noart": float(np.median(b)),
            "interpretation": significance
        }

        print(f"\n📊 Mann-Whitney U — Bias AOS (con artefacto vs. sin artefacto):")
        print(f"   Grupo CON artefacto (n={len(a)}):  "
              f"AOS media={np.mean(a):.4f} ± {np.std(a):.4f}")
        print(f"   Grupo SIN artefacto (n={len(b)}):  "
              f"AOS media={np.mean(b):.4f} ± {np.std(b):.4f}")
        print(f"   U = {u_stat:.1f}  |  p = {p_value:.4f}  ({significance})")
        print(f"   Effect size r = {effect_size_r:.3f} ({magnitude})")
        if significance != "No significativo":
            print(f"   → Sesgo Clever Hans ESTADÍSTICAMENTE SIGNIFICATIVO")
        else:
            print(f"   → No se detectó sesgo estadísticamente significativo")
            print(f"     (aumentar n de muestra o usar datos reales con artefactos naturales)")

        return result


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 4.1: TEST-TIME AUGMENTATION (TTA)
# ─────────────────────────────────────────────────────────────────────────────
class TestTimeAugmentation:
    """
    Test-Time Augmentation para mejorar la robustez de las predicciones.
    
    Aplica múltiples transformaciones geométricas a cada imagen de test y
    promedia las predicciones. Mejora típica: +1-3% AUC en dermoscopía.
    
    Transformaciones usadas:
        - Original
        - Horizontal flip
        - Vertical flip  
        - Rotación 90°
        - Rotación 180°
        - Rotación 270°
    
    Referencia:
        Matsunaga K et al. "Image Classification of Melanoma, Nevus and 
        Seborrheic Keratosis by Ensemble of CNNs" ISIC 2017 Challenge.
    """
    
    @staticmethod
    @torch.no_grad()
    def predict_with_tta(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        n_augmentations: int = 6
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ejecuta predicción con TTA.
        
        Args:
            model: Modelo entrenado.
            loader: DataLoader de test/val.
            device: Dispositivo.
            n_augmentations: Número de augmentaciones (default: 6).
        
        Returns:
            (y_true, y_prob_averaged): Labels y probabilidades promediadas.
        """
        model.eval()
        all_labels = []
        all_probs_accumulated = []
        
        from tqdm import tqdm
        for batch in tqdm(loader, desc="  TTA Inferencia", ncols=70, leave=False):
            images, labels = batch[0], batch[1]
            images = images.to(device, non_blocking=True)
            batch_probs = []
            
            # Transformaciones TTA
            augmentations = [
                lambda x: x,                          # Original
                lambda x: torch.flip(x, dims=[3]),     # Horizontal flip
                lambda x: torch.flip(x, dims=[2]),     # Vertical flip
                lambda x: torch.rot90(x, k=1, dims=[2, 3]),  # 90°
                lambda x: torch.rot90(x, k=2, dims=[2, 3]),  # 180°
                lambda x: torch.rot90(x, k=3, dims=[2, 3]),  # 270°
            ]
            
            for aug_fn in augmentations[:n_augmentations]:
                aug_images = aug_fn(images)
                logits = model(aug_images)
                if logits.dim() == 1 or (logits.dim() == 2 and logits.shape[1] == 1):
                    probs = torch.sigmoid(logits).view(-1)
                else:
                    probs = torch.softmax(logits, dim=1)[:, 1]
                batch_probs.append(probs.cpu().numpy())
            
            # Promediar probabilidades de todas las augmentaciones
            avg_probs = np.mean(batch_probs, axis=0)
            all_probs_accumulated.extend(avg_probs)
            all_labels.extend(labels.cpu().numpy())
        
        return np.array(all_labels, dtype=np.float64), np.array(all_probs_accumulated, dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 4.2: CALIBRACIÓN DE PROBABILIDADES — TEMPERATURE SCALING
# ─────────────────────────────────────────────────────────────────────────────
class TemperatureScaling:
    """
    Calibración de probabilidades mediante Temperature Scaling (Platt scaling).
    
    Aprende un parámetro T tal que: p_calibrada = σ(logit / T)
    
    Optimizado sobre el conjunto de VALIDACIÓN (nunca test).
    
    Referencia:
        Guo C et al. "On Calibration of Modern Neural Networks" ICML 2017.
    """
    
    def __init__(self):
        self.temperature = 1.0
    
    def fit(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 0.01,
        max_iter: int = 100
    ) -> float:
        """
        Optimiza la temperatura T sobre el conjunto de validación.
        
        Args:
            model: Modelo entrenado (en modo eval).
            val_loader: DataLoader de validación.
            device: Dispositivo.
            lr: Tasa de aprendizaje para optimización de T.
            max_iter: Iteraciones máximas.
        
        Returns:
            Temperatura óptima T.
        """
        model.eval()
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch[0], batch[1]
                images = images.to(device, non_blocking=True)
                logits = model(images)
                if logits.dim() == 2 and logits.shape[1] == 1:
                    logits = logits.view(-1)
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
        
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels).float()
        
        # Optimizar T minimizando BCE loss
        temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
        optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
        
        def eval_fn():
            optimizer.zero_grad()
            scaled_logits = all_logits / temperature
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                scaled_logits, all_labels
            )
            loss.backward()
            return loss
        
        optimizer.step(eval_fn)
        self.temperature = float(temperature.item())
        
        print(f"   🌡️  Temperature Scaling: T = {self.temperature:.4f}")
        return self.temperature
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Aplica temperature scaling a logits crudos."""
        return 1.0 / (1.0 + np.exp(-logits / self.temperature))


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 4.3: EVALUADOR CLÍNICO CON UMBRAL NO CONTAMINADO
# ─────────────────────────────────────────────────────────────────────────────
class ClinicalEvaluator:
    """
    Evaluador clínico con separación estricta entre calibración y evaluación.

    FLUJO OBLIGATORIO (previene fuga de datos):
    ─────────────────────────────────────────
    1. evaluator = ClinicalEvaluator(model, device)
    2. threshold = evaluator.calibrate_threshold(val_loader)   ← sobre VAL
    3. metrics   = evaluator.evaluate_test(test_loader, threshold) ← sobre TEST

    ❌ PROHIBIDO: calcular el umbral y las métricas en el mismo set (v1.0 bug).
    ✅ CORRECTO:  umbral desde VAL (donde el modelo NO vio durante test).
    """

    CLASS_NAMES = {0: "Benigno", 1: "Maligno"}

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        clinical_sensitivity_target: float = 0.90,
        threshold_objective: str = "sensitivity",
        clinical_precision_target: Optional[float] = None,
        minimum_recall_for_precision: float = 0.0,
    ):
        self.model    = model
        self.device   = device
        self.sensitivity_target = clinical_sensitivity_target
        self.threshold_objective = threshold_objective
        self.precision_target = clinical_precision_target
        self.minimum_recall_for_precision = minimum_recall_for_precision

        if self.threshold_objective not in {"sensitivity", "precision"}:
            raise ValueError(
                "threshold_objective debe ser 'sensitivity' o 'precision'."
            )
        if self.threshold_objective == "precision" and self.precision_target is None:
            raise ValueError(
                "clinical_precision_target es obligatorio cuando "
                "threshold_objective='precision'."
            )

        # Estado interno
        self._calibrated_threshold: Optional[float] = None
        self._val_y_true:  Optional[np.ndarray] = None
        self._val_y_prob:  Optional[np.ndarray] = None
        self._test_y_true: Optional[np.ndarray] = None
        self._test_y_prob: Optional[np.ndarray] = None

    @torch.no_grad()
    def _run_inference_internal(
        self, loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inferencia pura: retorna (y_true, y_prob). Sin efectos secundarios.
        Compatible con modelos de salida única (BCEWithLogitsLoss) y 2 clases (CE)."""
        self.model.eval()
        all_labels, all_probs = [], []

        from tqdm import tqdm
        for batch in tqdm(loader, desc="  Inferencia", ncols=70, leave=False):
            images, labels = batch[0], batch[1]
            images = images.to(self.device, non_blocking=True)
            logits = self.model(images)
            # Manejar salida única (sigmoid) o 2 clases (softmax)
            if logits.dim() == 1 or (logits.dim() == 2 and logits.shape[1] == 1):
                probs = torch.sigmoid(logits).view(-1)
            else:
                probs = torch.softmax(logits, dim=1)[:, 1]
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        return np.array(all_labels, dtype=np.float64), np.array(all_probs, dtype=np.float64)

    def calibrate_threshold(
        self,
        val_loader: DataLoader
    ) -> float:
        """
        PASO 1 (OBLIGATORIO): Calibrar el umbral sobre el conjunto de VALIDACIÓN.

        El umbral óptimo se define como el mínimo valor θ tal que:
            Sensibilidad(θ) ≥ sensitivity_target (default: 90%)
        Entre todos los θ que satisfacen esa condición, se elige el que
        maximiza la Especificidad (menor tasa de falsos positivos).

        Si ningún θ alcanza el target, se usa el índice de Youden:
            θ* = argmax TPR(θ) - FPR(θ)

        TRIPOD+AI ítem 10d: "Describe cómo se determinaron los umbrales de
        clasificación y si se usó un conjunto de datos diferente al de test."

        Args:
            val_loader: DataLoader del conjunto de VALIDACIÓN (nunca del test).

        Returns:
            Umbral calibrado θ* ∈ [0, 1].
        """
        print("⚙️  Calibrando umbral en VAL set (previene fuga de datos)...")
        self._val_y_true, self._val_y_prob = self._run_inference_internal(val_loader)

        if self.threshold_objective == "sensitivity":
            fpr, tpr, thresholds = roc_curve(self._val_y_true, self._val_y_prob)

            # Umbrales que alcanzan Sensibilidad ≥ target
            valid_mask = (tpr >= self.sensitivity_target)
            valid_idx  = np.where(valid_mask)[0]

            if len(valid_idx) > 0:
                best_idx = valid_idx[np.argmin(fpr[valid_idx])]
                self._calibrated_threshold = float(thresholds[best_idx])
                method_used = "Sensibilidad-óptima"
            else:
                # Fallback: Youden Index
                youden_idx = np.argmax(tpr - fpr)
                self._calibrated_threshold = float(thresholds[youden_idx])
                method_used = "Youden Index (target Sens no alcanzado en VAL)"
                warnings.warn(
                    f"⚠️  Sensibilidad ≥ {self.sensitivity_target:.0%} no alcanzada en VAL. "
                    f"Umbral por Youden: {self._calibrated_threshold:.4f}"
                )
        else:
            threshold_grid = np.unique(self._val_y_prob)[::-1]
            candidates = []
            for threshold in threshold_grid:
                y_pred = (self._val_y_prob >= threshold).astype(int)
                predicted_positive = int(y_pred.sum())
                precision = precision_score(
                    self._val_y_true, y_pred, pos_label=1, zero_division=0
                )
                recall = recall_score(
                    self._val_y_true, y_pred, pos_label=1, zero_division=0
                )
                specificity = recall_score(
                    self._val_y_true, y_pred, pos_label=0, zero_division=0
                )
                candidates.append(
                    {
                        "threshold": float(threshold),
                        "precision": float(precision),
                        "recall": float(recall),
                        "specificity": float(specificity),
                        "predicted_positive": predicted_positive,
                    }
                )

            valid_candidates = [
                candidate for candidate in candidates
                if candidate["predicted_positive"] > 0
                and candidate["precision"] >= self.precision_target
                and candidate["recall"] >= self.minimum_recall_for_precision
            ]

            if valid_candidates:
                best_candidate = max(
                    valid_candidates,
                    key=lambda item: (
                        item["recall"],
                        item["specificity"],
                        item["precision"],
                    ),
                )
                self._calibrated_threshold = best_candidate["threshold"]
                method_used = (
                    f"Precisión-óptima (PPV ≥ {self.precision_target:.0%}, "
                    f"Recall ≥ {self.minimum_recall_for_precision:.0%})"
                )
            else:
                non_empty_candidates = [
                    candidate for candidate in candidates
                    if candidate["predicted_positive"] > 0
                ]
                best_candidate = max(
                    non_empty_candidates,
                    key=lambda item: (
                        item["precision"],
                        item["recall"],
                        item["specificity"],
                    ),
                )
                self._calibrated_threshold = best_candidate["threshold"]
                method_used = "Máxima precisión alcanzable (target PPV no alcanzado)"
                warnings.warn(
                    f"⚠️  Precision ≥ {self.precision_target:.0%} no alcanzada en VAL. "
                    f"Se usa el umbral de máxima PPV observada: "
                    f"{self._calibrated_threshold:.4f}"
                )

        # Métricas sobre VAL con el umbral calibrado
        y_pred_val = (self._val_y_prob >= self._calibrated_threshold).astype(int)
        sens_val = recall_score(self._val_y_true, y_pred_val, pos_label=1, zero_division=0)
        spec_val = recall_score(self._val_y_true, y_pred_val, pos_label=0, zero_division=0)
        prec_val = precision_score(self._val_y_true, y_pred_val, pos_label=1, zero_division=0)

        print(f"   Método: {method_used}")
        print(f"   θ* = {self._calibrated_threshold:.4f}")
        print(f"   Sensibilidad VAL @ θ*: {sens_val:.4f}")
        print(f"   Especificidad VAL @ θ*: {spec_val:.4f}")
        print(f"   Precision VAL @ θ*:    {prec_val:.4f}")
        print("   ✅ Umbral frozen — listo para aplicar al TEST set.")

        return self._calibrated_threshold

    def evaluate_test(
        self,
        test_loader: DataLoader,
        threshold: Optional[float] = None
    ) -> Dict:
        """
        PASO 2: Evalúa el modelo sobre el TEST set con umbral frozen.

        Args:
            test_loader: DataLoader del conjunto de TEST.
            threshold:   Umbral a usar. Si None, usa el calibrado en calibrate_threshold().
                         Nunca debe calcularse sobre test_loader.

        Raises:
            RuntimeError: Si no se ha llamado calibrate_threshold() y threshold=None.

        Returns:
            Dict con todas las métricas clínicas.
        """
        if threshold is not None:
            operative_threshold = threshold
        elif self._calibrated_threshold is not None:
            operative_threshold = self._calibrated_threshold
        else:
            raise RuntimeError(
                "❌ FUGA DE DATOS PREVENIDA: No hay umbral calibrado.\n"
                "   Llamar primero: evaluator.calibrate_threshold(val_loader)\n"
                "   El umbral NUNCA debe calcularse sobre el test set."
            )

        print(f"\n🔍 Evaluando TEST set con θ_frozen = {operative_threshold:.4f}...")
        self._test_y_true, self._test_y_prob = self._run_inference_internal(test_loader)

        return self._compute_metrics(
            self._test_y_true, self._test_y_prob, operative_threshold
        )

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float
    ) -> Dict:
        """Calcula métricas clínicas dado un umbral frozen."""
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv         = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1          = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        acc         = accuracy_score(y_true, y_pred)

        try:
            auc_roc = roc_auc_score(y_true, y_prob)
            auc_pr  = average_precision_score(y_true, y_prob)
        except Exception:
            auc_roc = auc_pr = 0.5

        metrics = {
            "AUC-ROC": auc_roc, "AUC-PR": auc_pr, "Accuracy": acc,
            "Sensitivity": sensitivity, "Specificity": specificity,
            "Precision": ppv, "NPV": npv, "F1-Score": f1,
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
            "Threshold": float(threshold),
            # Guardar arrays para tests estadísticos posteriores
            "_y_true": y_true, "_y_prob": y_prob
        }

        print("\n" + "═" * 55)
        print("  RESULTADOS EVALUACIÓN TEST (umbral desde VAL, frozen)")
        print("═" * 55)
        print(f"  AUC-ROC:          {auc_roc:.4f}")
        print(f"  AUC-PR:           {auc_pr:.4f}")
        print(f"  Accuracy:         {acc:.4f}")
        print(f"  ─────────────────────────────────────────────")
        print(f"  Sensitivity:      {sensitivity:.4f}  ⬅ PRIMARIA")
        print(f"  Specificity:      {specificity:.4f}")
        print(f"  Precision (PPV):  {ppv:.4f}")
        print(f"  NPV:              {npv:.4f}")
        print(f"  F1-Score:         {f1:.4f}")
        print(f"  ─────────────────────────────────────────────")
        print(f"  TP={tp} | FP={fp} | FN={fn} | TN={tn}")
        print(f"  Umbral (frozen de VAL): {threshold:.4f}")
        print("═" * 55)

        return metrics

    def compute_bootstrap_cis(
        self,
        metrics: Dict,
        n_bootstrap: int = 2000
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Calcula IC BCa del 95% para todas las métricas clave.

        Returns:
            Dict: {metric_name: (estimate, ci_lower, ci_upper)}
        """
        y_true = metrics["_y_true"]
        y_prob = metrics["_y_prob"]
        threshold = metrics["Threshold"]

        print(f"\n📊 Calculando Intervalos de Confianza BCa ({n_bootstrap} remuestras)...")
        cis = {}
        for metric in ["auc_roc", "sensitivity", "specificity", "f1", "ppv"]:
            est, lo, hi = StatisticalValidator.bootstrap_ci(
                y_true, y_prob, metric=metric,
                threshold=threshold, n_bootstrap=n_bootstrap
            )
            cis[metric] = (est, lo, hi)
            label = metric.replace("_", " ").upper()
            print(f"  {label:<15}: {est:.4f}  [95% IC BCa: {lo:.4f} – {hi:.4f}]")

        return cis

    # ─────────────────────────────────────────────────────────────────────
    # Visualización completa (heredada de v1.0 con correcciones menores)
    # ─────────────────────────────────────────────────────────────────────
    def plot_comprehensive_evaluation(
        self,
        metrics: Dict,
        cis: Optional[Dict] = None,
        history: Optional[Dict] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Genera figura maestra de evaluación clínica.
        Incluye IC BCa si se pasan en cis=evaluator.compute_bootstrap_cis(metrics).
        """
        PALETTE = {
            "primary": "#00d4ff", "secondary": "#ff6b9d",
            "success": "#44ff88", "warning": "#ffd166",
            "danger": "#ff4444", "neutral": "#8899aa",
            "bg": "#0d1117", "surface": "#161b22", "grid": "#21262d"
        }

        plt.style.use("dark_background")
        y_true = metrics["_y_true"]
        y_prob = metrics["_y_prob"]
        threshold = metrics["Threshold"]

        has_history = history is not None and len(history.get("train_loss", [])) > 0
        n_rows = 3 if has_history else 2
        fig = plt.figure(figsize=(20, 7 * n_rows))
        fig.patch.set_facecolor(PALETTE["bg"])
        gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.45, wspace=0.35)

        # ── Subplot 1: Curva ROC ──────────────────────────────────────────
        ax_roc = fig.add_subplot(gs[0, 0])
        ax_roc.set_facecolor(PALETTE["surface"])
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = metrics["AUC-ROC"]
        n_pts = len(fpr)
        colors_roc = plt.cm.plasma(np.linspace(0.2, 0.9, max(n_pts - 1, 1)))
        for i in range(min(n_pts - 1, len(colors_roc))):
            ax_roc.plot(fpr[i:i+2], tpr[i:i+2], color=colors_roc[i], linewidth=2.5)
        ax_roc.plot([0, 1], [0, 1], '--', color=PALETTE["neutral"], alpha=0.5, linewidth=1.5)
        ax_roc.fill_between(fpr, tpr, alpha=0.12, color=PALETTE["primary"])
        op_fpr = 1 - metrics["Specificity"]
        op_tpr = metrics["Sensitivity"]
        ax_roc.scatter(op_fpr, op_tpr, s=150, zorder=5, color=PALETTE["danger"],
                      edgecolors='white', linewidth=2,
                      label=f"θ*={threshold:.3f}\n(θ de VAL frozen)")

        # AUC con IC si disponible
        auc_label = f"AUC = {auc_val:.4f}"
        if cis and "auc_roc" in cis:
            _, lo, hi = cis["auc_roc"]
            auc_label += f"\n[95% IC: {lo:.4f}–{hi:.4f}]"
        ax_roc.set_title(f"Curva ROC\n{auc_label}", color='white',
                        fontsize=11, fontweight='bold')
        ax_roc.set_xlim(-0.02, 1.02); ax_roc.set_ylim(-0.02, 1.02)
        ax_roc.set_xlabel("1 - Especificidad", color='#ccc', fontsize=9)
        ax_roc.set_ylabel("Sensibilidad", color='#ccc', fontsize=9)
        ax_roc.legend(loc="lower right", fontsize=8,
                     facecolor=PALETTE["surface"], edgecolor='#555')
        ax_roc.grid(alpha=0.2, color=PALETTE["grid"]); ax_roc.tick_params(colors='#ccc')

        # ── Subplot 2: Curva PR ───────────────────────────────────────────
        ax_pr = fig.add_subplot(gs[0, 1])
        ax_pr.set_facecolor(PALETTE["surface"])
        prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)
        ax_pr.plot(rec_curve, prec_curve, color=PALETTE["secondary"], linewidth=2.5)
        ax_pr.fill_between(rec_curve, prec_curve, alpha=0.15, color=PALETTE["secondary"])
        prevalence = float(y_true.mean())
        ax_pr.axhline(prevalence, linestyle='--', color=PALETTE["neutral"],
                     alpha=0.6, linewidth=1.5, label=f"Prevalencia ({prevalence:.2f})")
        ax_pr.scatter(metrics["Sensitivity"], metrics["Precision"], s=150,
                     zorder=5, color=PALETTE["danger"], edgecolors='white', linewidth=2)
        auc_pr_label = f"AUC-PR = {metrics['AUC-PR']:.4f}"
        ax_pr.set_title(f"Curva Precisión-Recall\n{auc_pr_label}",
                       color='white', fontsize=11, fontweight='bold')
        ax_pr.set_xlim(-0.02, 1.02); ax_pr.set_ylim(-0.02, 1.02)
        ax_pr.set_xlabel("Recall (Sensibilidad)", color='#ccc', fontsize=9)
        ax_pr.set_ylabel("Precisión", color='#ccc', fontsize=9)
        ax_pr.legend(fontsize=8, facecolor=PALETTE["surface"], edgecolor='#555')
        ax_pr.grid(alpha=0.2, color=PALETTE["grid"]); ax_pr.tick_params(colors='#ccc')

        # ── Subplot 3: Distribución de probabilidades ─────────────────────
        ax_dist = fig.add_subplot(gs[0, 2])
        ax_dist.set_facecolor(PALETTE["surface"])
        bins = np.linspace(0, 1, 30)
        ax_dist.hist(y_prob[y_true == 0], bins=bins, alpha=0.65,
                    color=PALETTE["primary"], label="Benigno", density=True)
        ax_dist.hist(y_prob[y_true == 1], bins=bins, alpha=0.65,
                    color=PALETTE["secondary"], label="Maligno", density=True)
        ax_dist.axvline(threshold, color=PALETTE["danger"], linestyle='--',
                       linewidth=2, label=f"θ*={threshold:.3f} (desde VAL)")
        ax_dist.set_xlabel("P(Maligno) predicho", color='#ccc', fontsize=9)
        ax_dist.set_ylabel("Densidad", color='#ccc', fontsize=9)
        ax_dist.set_title("Distribución de Probabilidades\n(Calibración visual)",
                          color='white', fontsize=11, fontweight='bold')
        ax_dist.legend(fontsize=8, facecolor=PALETTE["surface"], edgecolor='#555')
        ax_dist.grid(alpha=0.2, color=PALETTE["grid"]); ax_dist.tick_params(colors='#ccc')

        # ── Subplot 4: Matriz de Confusión ────────────────────────────────
        ax_cm = fig.add_subplot(gs[1, 0])
        ax_cm.set_facecolor(PALETTE["surface"])
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cmap_custom = LinearSegmentedColormap.from_list(
            "clinical", ["#0d1117", "#00b4d8", "#00d4ff"]
        )
        sns.heatmap(cm_norm, annot=False, cmap=cmap_custom, ax=ax_cm,
                   cbar=True, linewidths=2, linecolor=PALETTE["bg"], vmin=0, vmax=1)
        for i, (row_data, row_cm) in enumerate(zip(cm_norm, cm)):
            for j, (pct, count) in enumerate(zip(row_data, row_cm)):
                labels_pos = [["TN", "FP"], ["FN", "TP"]]
                hc = [[PALETTE["success"], PALETTE["danger"]],
                      [PALETTE["danger"],  PALETTE["success"]]]
                ax_cm.text(j + 0.5, i + 0.35, f"{labels_pos[i][j]}\n{count}",
                          ha='center', va='center', fontsize=13,
                          fontweight='bold', color=hc[i][j])
                ax_cm.text(j + 0.5, i + 0.72, f"({pct:.1%})",
                          ha='center', va='center', fontsize=9, color='#ccc')
        ax_cm.set_xticklabels(["Pred: Benigno", "Pred: Maligno"],
                              color='#ccc', fontsize=9)
        ax_cm.set_yticklabels(["Real: Benigno", "Real: Maligno"],
                              color='#ccc', fontsize=9, rotation=0)
        ax_cm.set_title(f"Matriz de Confusión\nθ*={threshold:.4f} (frozen de VAL)",
                       color='white', fontsize=11, fontweight='bold')

        # ── Subplot 5: Barras de métricas con IC ─────────────────────────
        ax_bar = fig.add_subplot(gs[1, 1])
        ax_bar.set_facecolor(PALETTE["surface"])
        metric_names = ["AUC-ROC", "Sensitivity", "Specificity", "Precision",
                        "F1-Score", "Accuracy"]
        metric_keys_ci = ["auc_roc", "sensitivity", "specificity", "ppv", "f1", None]
        metric_values = [metrics[m] for m in metric_names]

        bar_colors = []
        for name, val in zip(metric_names, metric_values):
            if name == "Sensitivity":
                bar_colors.append(PALETTE["success"] if val >= 0.9 else PALETTE["warning"])
            elif val >= 0.8: bar_colors.append(PALETTE["primary"])
            elif val >= 0.6: bar_colors.append(PALETTE["warning"])
            else: bar_colors.append(PALETTE["danger"])

        bars = ax_bar.barh(metric_names, metric_values, color=bar_colors,
                          height=0.6, edgecolor='none')

        # Añadir barras de error (IC BCa) si disponibles
        if cis:
            for i, (name, key) in enumerate(zip(metric_names, metric_keys_ci)):
                if key and key in cis:
                    est, lo, hi = cis[key]
                    ax_bar.errorbar(est, i, xerr=[[est - lo], [hi - est]],
                                   fmt='none', color='white', capsize=4,
                                   linewidth=1.5, capthick=1.5)

        for bar, val, name in zip(bars, metric_values, metric_names):
            ax_bar.text(min(val + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
                       f"{val:.4f}",
                       va='center', fontsize=9,
                       color=PALETTE["warning"] if name == "Sensitivity" else 'white',
                       fontweight='bold' if name == "Sensitivity" else 'normal')
        ax_bar.axvline(0.9, color=PALETTE["danger"], linestyle='--',
                      alpha=0.5, linewidth=1.5)
        ax_bar.set_xlim(0, 1.15)
        ax_bar.set_xlabel("Valor", color='#ccc', fontsize=9)
        ax_bar.set_title("Métricas con 95% IC BCa\n(barras de error)",
                        color='white', fontsize=11, fontweight='bold')
        ax_bar.tick_params(colors='#ccc')
        ax_bar.grid(axis='x', alpha=0.2, color=PALETTE["grid"])

        # ── Subplot 6: Tabla resumen ──────────────────────────────────────
        ax_t = fig.add_subplot(gs[1, 2])
        ax_t.set_facecolor(PALETTE["surface"]); ax_t.axis('off')
        rows_data = [
            ["Métrica", "Valor", "IC 95%"],
            ["AUC-ROC",     f"{metrics['AUC-ROC']:.4f}",
             f"[{cis['auc_roc'][1]:.3f}–{cis['auc_roc'][2]:.3f}]" if cis and "auc_roc" in cis else "—"],
            ["Sensitivity", f"{metrics['Sensitivity']:.4f}",
             f"[{cis['sensitivity'][1]:.3f}–{cis['sensitivity'][2]:.3f}]" if cis and "sensitivity" in cis else "—"],
            ["Specificity", f"{metrics['Specificity']:.4f}",
             f"[{cis['specificity'][1]:.3f}–{cis['specificity'][2]:.3f}]" if cis and "specificity" in cis else "—"],
            ["F1-Score",    f"{metrics['F1-Score']:.4f}",
             f"[{cis['f1'][1]:.3f}–{cis['f1'][2]:.3f}]" if cis and "f1" in cis else "—"],
            ["FN (miss)",   f"{metrics['FN']}", "Melanomas no detectados"],
            ["θ* (VAL)",    f"{metrics['Threshold']:.4f}", "Umbral frozen"],
        ]
        for ri, row in enumerate(rows_data):
            y = 0.95 - ri * 0.13
            bg = "#1e3a5f" if ri == 0 else ("#0d1117" if ri % 2 == 0 else "#161b22")
            ax_t.axhspan(y - 0.13, y, color=bg, alpha=0.8)
            for ci_col, (cell, xw) in enumerate(zip(row, [0.3, 0.22, 0.42])):
                xpos = sum([0.3, 0.22, 0.42][:ci_col]) + xw / 2
                c = 'white' if ri == 0 else (PALETTE["warning"] if ri == 2 and ci_col == 1
                                             else PALETTE["primary"] if ci_col == 1 else '#ccc')
                ax_t.text(xpos, y - 0.065, cell, ha='center', va='center',
                         color=c, fontsize=8,
                         fontweight='bold' if ri == 0 else 'normal',
                         transform=ax_t.transAxes)
        ax_t.set_title("Resumen Estadístico", color='white', fontsize=11, fontweight='bold')
        ax_t.set_xlim(0, 1); ax_t.set_ylim(0, 1)

        # ── Subplots de convergencia (si hay historial) ───────────────────
        if has_history:
            epochs = list(range(1, len(history["train_loss"]) + 1))

            ax_loss = fig.add_subplot(gs[2, 0])
            ax_loss.set_facecolor(PALETTE["surface"])
            ax_loss.plot(epochs, history["train_loss"], color=PALETTE["primary"],
                        linewidth=2.5, label="Train Loss")
            ax_loss.plot(epochs, history["val_loss"], color=PALETTE["secondary"],
                        linewidth=2.5, label="Val Loss")
            ax_loss.fill_between(epochs, history["train_loss"], history["val_loss"],
                                alpha=0.1, color=PALETTE["warning"])
            if "val_auc" in history:
                best_ep = int(np.argmax(history["val_auc"])) + 1
                ax_loss.axvline(best_ep, color=PALETTE["success"], linestyle=':',
                               linewidth=2, label=f"Mejor (ep.{best_ep})")
            ax_loss.set_xlabel("Época", color='#ccc', fontsize=9)
            ax_loss.set_ylabel("Focal Loss", color='#ccc', fontsize=9)
            ax_loss.set_title("Convergencia: Loss", color='white', fontsize=11, fontweight='bold')
            ax_loss.legend(fontsize=8, facecolor=PALETTE["surface"], edgecolor='#555')
            ax_loss.grid(alpha=0.2, color=PALETTE["grid"]); ax_loss.tick_params(colors='#ccc')

            ax_auc = fig.add_subplot(gs[2, 1])
            ax_auc.set_facecolor(PALETTE["surface"])
            if "val_auc" in history:
                ax_auc.plot(epochs, history["val_auc"], color=PALETTE["success"],
                           linewidth=2.5, label="Val AUC-ROC")
                best_auc = max(history["val_auc"])
                ax_auc.axhline(best_auc, color=PALETTE["warning"], linestyle='--',
                              linewidth=1.5, alpha=0.7, label=f"Best: {best_auc:.4f}")
                ax_auc.fill_between(epochs, history["val_auc"], alpha=0.12,
                                   color=PALETTE["success"])
            ax_auc.set_ylim(0.4, 1.05)
            ax_auc.set_xlabel("Época", color='#ccc', fontsize=9)
            ax_auc.set_ylabel("AUC-ROC", color='#ccc', fontsize=9)
            ax_auc.set_title("Convergencia: AUC Validación",
                            color='white', fontsize=11, fontweight='bold')
            ax_auc.legend(fontsize=8, facecolor=PALETTE["surface"], edgecolor='#555')
            ax_auc.grid(alpha=0.2, color=PALETTE["grid"]); ax_auc.tick_params(colors='#ccc')

            ax_lr = fig.add_subplot(gs[2, 2])
            ax_lr.set_facecolor(PALETTE["surface"])
            if history.get("lr_head"):
                ax_lr.plot(epochs, history["lr_head"], color=PALETTE["primary"],
                          linewidth=2.5, label="LR — Cabeza")
            if history.get("lr_backbone"):
                ax_lr.plot(epochs, history["lr_backbone"], color=PALETTE["secondary"],
                          linewidth=2, linestyle='--', label="LR — Backbone", alpha=0.8)
            ax_lr.set_yscale("log")
            ax_lr.set_xlabel("Época", color='#ccc', fontsize=9)
            ax_lr.set_ylabel("LR (log)", color='#ccc', fontsize=9)
            ax_lr.set_title("CosineAnnealingLR Schedule",
                           color='white', fontsize=11, fontweight='bold')
            ax_lr.legend(fontsize=8, facecolor=PALETTE["surface"], edgecolor='#555')
            ax_lr.grid(alpha=0.2, color=PALETTE["grid"]); ax_lr.tick_params(colors='#ccc')

        fig.suptitle(
            "Evaluación Clínica — EfficientNet-B4 | Focal Loss (γ=2) | TTA | Protocolo TRIPOD+AI\n"
            "Umbral θ* calibrado en VAL (frozen para TEST) | IC BCa 95%",
            color='white', fontsize=13, fontweight='bold', y=1.01
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor=PALETTE["bg"])
            print(f"✅ Figura guardada: {save_path}")

        return fig

    # ─────────────────────────────────────────────────────────────────────
    # MITIGACIÓN LEGACY (solo inferencia — documentada como ablación, no mitigación real)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def get_inference_crop_transform(crop_fraction: float = 0.85, img_size: int = 224):
        """
        Transformación de recorte central para ablación de inferencia.

        ⚠️  DOCUMENTACIÓN DE LIMITACIÓN:
            Esta es una ablación de inferencia (sin reentrenamiento), NO una
            mitigación real del sesgo Clever Hans. El modelo ha aprendido las
            representaciones biasadas durante training; cambiar el preprocesamiento
            en inferencia no deshace ese aprendizaje.

            Para mitigación real: usar MaskedISICDataset durante TRAINING
            (src/data_pipeline.py, COND-C).

            Este método se mantiene para comparación ablativa. Etiquetar siempre
            como "Ablación de inferencia" en papers, nunca como "modelo mitigado".
        """
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        crop_size = int(img_size * crop_fraction)
        return A.Compose([
            A.CenterCrop(height=crop_size, width=crop_size),
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


if __name__ == "__main__":
    print("Módulo evaluation.py v2.0 cargado.")
    print("Flujo correcto:")
    print("  evaluator = ClinicalEvaluator(model, device)")
    print("  threshold = evaluator.calibrate_threshold(val_loader)  # ← VAL")
    print("  metrics   = evaluator.evaluate_test(test_loader)        # ← TEST")
    print("  cis       = evaluator.compute_bootstrap_cis(metrics)")
    print("  delong    = StatisticalValidator.delong_test(y_true, prob_A, prob_B)")
    print("  mw        = StatisticalValidator.mann_whitney_bias(aos_with, aos_without)")
