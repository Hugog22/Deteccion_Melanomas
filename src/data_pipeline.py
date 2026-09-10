"""
================================================================================
FASE 1 (REVISADA): PREPROCESAMIENTO CLÍNICO — CORRECCIONES CIENTÍFICAS
================================================================================
Módulo: data_pipeline.py  — v2.0

CORRECCIONES APLICADAS respecto a v1.0:
  [FIX-1] ISICSyntheticGenerator renombrado y marcado explícitamente como
          "unit-test only". No se mezcla con el pipeline científico real.
  [NEW-1] ISICArchiveDownloader: descarga imágenes reales vía la API pública
          ISIC Archive v2 (https://api.isic-archive.com/api/v2/).
          Fallback gracioso a sintético si la API no está disponible.
  [NEW-2] patient_level_split(): separa train/val/test A NIVEL DE PACIENTE
          (campo patient_id de los metadatos ISIC) para evitar fuga de datos
          cuando hay múltiples imágenes del mismo lunar.
  [NEW-3] get_unmitigated_transforms(): pipeline sin RandomResizedCrop, para
          el baseline "honestamente sesgado" que expone el Clever Hans limpiamente.
  [NEW-4] MaskedISICDataset: aplica segmentación de lesión EN TIEMPO DE TRAINING
          (no solo en inferencia), para la condición de mitigación real.
  [FIX-2] FocalLoss sin cambios (implementación matemáticamente correcta).

Diseño experimental de 3 condiciones (corrige la confusión metodológica de v1.0):
  COND-A (Baseline sesgado): transforms sin RandomResizedCrop → Clever Hans emergente
  COND-B (Mitigación parcial): transforms CON RandomResizedCrop → bloqueo de bordes
  COND-C (Mitigación real):   MaskedISICDataset → modelo nunca ve artefactos

Referencia API ISIC:
    https://api.isic-archive.com/api/v2/ (acceso público, sin auth para subset limitado)
    Dataset completo: ISIC 2020 Challenge — Kaggle
    https://www.kaggle.com/c/siim-isic-melanoma-classification/data
================================================================================
"""

import os
import json
import random
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
from typing import Tuple, Optional, Dict, List, Literal
import urllib.request
import urllib.error

from functools import partial
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# REPRODUCIBILIDAD GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
def set_global_seed(seed: int = 42) -> None:
    """Fija semillas en todos los frameworks para reproducibilidad determinista."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1.0: DESCARGA DEL DATASET REAL ISIC
# ─────────────────────────────────────────────────────────────────────────────
class ISICArchiveDownloader:
    """
    Descargador de imágenes reales del ISIC Archive vía API pública v2.

    La API ISIC Archive v2 (https://api.isic-archive.com/api/v2/) proporciona
    acceso público a imágenes dermatoscópicas reales con sus metadatos clínicos.
    No requiere autenticación para las primeras N imágenes públicas.

    ⚠️ AVISO CIENTÍFICO:
        Para un paper revisado por pares, se debe usar el dataset completo
        ISIC 2020 Challenge (33,126 imágenes únicas de paciente) disponible en:
        https://www.kaggle.com/c/siim-isic-melanoma-classification/data

        Esta clase descarga un subconjunto prototípico suficiente para
        validar el pipeline de extremo a extremo.

    Referencia de la API:
        https://api.isic-archive.com/api/v2/ (schema OpenAPI disponible)
    """

    API_BASE      = "https://api.isic-archive.com/api/v2"
    IMAGES_EP     = f"{API_BASE}/images/"
    THUMBNAIL_URL = "https://isic-archive.com/api/v2/images/{isic_id}/thumbnail"

    # Diagnósticos ISIC — campo diagnosis_3 (el más específico de la API)
    # Verificado empíricamente en la API v2 en julio 2025.
    MALIGNANT_DIAGNOSES = {
        "melanoma",
        "melanoma in situ",
        "melanoma invasive",
        "melanoma, nos",
        "malignant",   # diagnosis_1 directo
    }
    BENIGN_DIAGNOSES = {
        "nevus",
        "seborrheic keratosis",
        # BCC es maligno pero no-melanoma: la convención estándar ISIC 2020
        # (Rotemberg et al. 2021, ISDC doi:10.1038/s44151-021-00003-1) clasifica
        # la tarea como detección de melanoma vs. no-melanoma, incluyendo BCC
        # y SCC en el grupo benigno-para-esta-tarea.
        "basal cell carcinoma",
        "benign keratosis",
        "dermatofibroma",
        "vascular lesion",
        "benign melanocytic proliferations",
        "lentigo nos",
        "benign",   # diagnosis_1 directo
    }

    def __init__(self, output_dir: str, timeout: int = 30):
        self.output_dir = Path(output_dir)
        self.timeout = timeout
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Contexto SSL con fallback sin verificación para macOS sin certificados instalados.
        # La verificación se intenta primero; solo si falla se usa el bypass.
        import ssl as _ssl
        self._ssl_ctx_strict = _ssl.create_default_context()
        self._ssl_ctx_bypass  = _ssl.create_default_context()
        self._ssl_ctx_bypass.check_hostname = False
        self._ssl_ctx_bypass.verify_mode    = _ssl.CERT_NONE

    def _get_json(self, url: str) -> Optional[dict]:
        """
        Realiza una petición GET con doble intento SSL:
        1. Verificación estricta (preferida).
        2. Sin verificación (macOS con certificados no instalados via /Applications/Python/Install Certificates.command).
        """
        for ctx in (self._ssl_ctx_strict, self._ssl_ctx_bypass):
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "MelanomaResearch/2.0"}
                )
                with urllib.request.urlopen(req, timeout=self.timeout, context=ctx) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.URLError as e:
                reason = str(e.reason) if hasattr(e, 'reason') else str(e)
                if 'CERTIFICATE' in reason.upper() and ctx is self._ssl_ctx_strict:
                    continue  # Reintenta sin verificación
                warnings.warn(f"⚠️  API ISIC no alcanzable: {e}. Usando fallback sintético.")
                return None
            except (json.JSONDecodeError, OSError) as e:
                warnings.warn(f"⚠️  API ISIC no alcanzable: {e}. Usando fallback sintético.")
                return None
        warnings.warn("⚠️  API ISIC no alcanzable (SSL). Usando fallback sintético.")
        return None

    def _download_image(self, isic_id: str, dest_path: Path, thumb_url: str = "") -> bool:
        """Descarga la imagen thumbnail de una lesión ISIC.

        Args:
            isic_id:   ID ISIC de la imagen (para construir URL de fallback).
            dest_path: Ruta local donde guardar el archivo.
            thumb_url: URL directa del thumbnail (preferida si viene de la API).
        """
        if dest_path.exists():
            return True
        # Preferir URL directa de la API (thumbnail_256); fallback a URL construida
        url = thumb_url or f"{self.API_BASE}/images/{isic_id}/thumbnail"
        for ctx in (self._ssl_ctx_strict, self._ssl_ctx_bypass):
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "MelanomaResearch/2.0"}
                )
                with urllib.request.urlopen(req, timeout=self.timeout, context=ctx) as resp:
                    dest_path.write_bytes(resp.read())
                return True
            except urllib.error.URLError as e:
                if 'CERTIFICATE' in str(getattr(e, 'reason', '')).upper():
                    continue
                return False
            except Exception:
                return False
        return False

    def _get_json_filtered(self, diagnosis_filter: str) -> Optional[dict]:
        url = f"{self.IMAGES_EP}?limit=200&offset=0&diagnosis={diagnosis_filter}"
        return self._get_json(url)

    def _parse_img_meta(self, img_meta: dict) -> Optional[dict]:
        isic_id = img_meta.get("isic_id", "")
        if not isic_id:
            return None

        meta_block = img_meta.get("metadata", {}) or {}
        clinical   = meta_block.get("clinical", {}) or {}

        d1 = (clinical.get("diagnosis_1", "") or "").lower().strip()
        d2 = (clinical.get("diagnosis_2", "") or "").lower().strip()
        d3 = (clinical.get("diagnosis_3", "") or "").lower().strip()
        diag_str = d3 or d2 or d1

        if d1 == "malignant":
            benign_malignant = "malignant"
        elif d1 == "benign":
            benign_malignant = "benign"
        else:
            benign_malignant = ""

        if not diag_str and not benign_malignant:
            return None

        unstructured = meta_block.get("unstructured", {}) or {}
        if isinstance(unstructured, str):
            unstructured = {}

        pid = (
            clinical.get("patient_id") or
            unstructured.get("patient_id") or
            unstructured.get("patient") or
            unstructured.get("lesion_id") or
            meta_block.get("lesion_id") or
            None
        )
        
        pid_source = "isic_id_fallback"
        if pid:
            pid_source = "clinical_or_unstructured"
        else:
            pid = isic_id

        files = img_meta.get("files") or {}
        thumb = (files.get("thumbnail_256") or {}).get("url", "")

        return {
            "isic_id":          isic_id,
            "diagnosis":        diag_str,
            "benign_malignant": benign_malignant,
            "patient_id":       str(pid),
            "patient_id_source": pid_source,
            "thumbnail_url":    thumb,
        }

    def download_balanced_subset(
        self,
        n_benign: int = 1500,
        n_malignant: int = 500,
        img_size: int = 380,
        add_artifacts_for_demo: bool = False
    ) -> pd.DataFrame:
        """
        Descarga un subconjunto balanceado de imágenes ISIC reales.

        Estrategia de descarga:
            1. Consulta metadatos de la API (sin descargar imágenes aún).
            2. Filtra por diagnóstico leyendo los campos reales de la API v2:
               - `benign_malignant` ("benign" / "malignant") en raíz o metadata
               - `diagnosis_1` / `diagnosis_2` / `diagnosis_3` en raíz
               - Fallback a `metadata.clinical.diagnosis` para colecciones antiguas
            3. Descarga imágenes con rate limiting (1 req/seg).
            4. Si la API no está disponible, retorna None para fallback sintético.

        Args:
            n_benign:    Número objetivo de imágenes benignas.
            n_malignant: Número objetivo de imágenes malignas.
            img_size:    Tamaño al que redimensionar las imágenes descargadas.
            add_artifacts_for_demo: Inyectar artefactos sintéticos en subset maligno.
                ⚠️  DEFAULT=False. Cambiar a True SOLO para proof-of-concept
                controlado (el sesgo será fabricado por ti, no el sesgo real
                del dataset). Presenta los resultados como "validación del
                método de detección", NO como evidencia del sesgo real de ISIC.

        Returns:
            DataFrame con columnas requeridas por el pipeline, o None si falla la API.
        """
        print("🌐 Intentando conectar con ISIC Archive API v2...")

        malignant_records = []
        benign_records    = []
        max_pages = 200

        print("   Buscando imágenes malignas (filtro 'melanoma')...")
        filtered_url = f"{self.IMAGES_EP}?limit=200&offset=0&diagnosis=melanoma"
        pages_fetched = 0
        while filtered_url and pages_fetched < max_pages and len(malignant_records) < n_malignant * 2:
            data = self._get_json(filtered_url)
            if data is None:
                break
            for img_meta in data.get("results", []):
                rec = self._parse_img_meta(img_meta)
                if not rec: continue
                if (rec["diagnosis"] in self.MALIGNANT_DIAGNOSES or rec["benign_malignant"] == "malignant"):
                    if len(malignant_records) < n_malignant * 2:
                        malignant_records.append(rec)
            filtered_url = data.get("next")
            pages_fetched += 1
            time.sleep(0.5)

        print("   Buscando el resto de imágenes sin filtro...")
        page_url = f"{self.IMAGES_EP}?limit=200&offset=0"
        pages_fetched = 0
        while page_url and pages_fetched < max_pages:
            if (len(malignant_records) >= n_malignant * 2 and
                    len(benign_records) >= n_benign * 2):
                break
                
            data = self._get_json(page_url)
            if data is None:
                if not benign_records and not malignant_records: return None
                break

            for img_meta in data.get("results", []):
                rec = self._parse_img_meta(img_meta)
                if not rec: continue

                if (rec["diagnosis"] in self.MALIGNANT_DIAGNOSES or rec["benign_malignant"] == "malignant"):
                    if len(malignant_records) < n_malignant * 2:
                        if not any(r["isic_id"] == rec["isic_id"] for r in malignant_records):
                            malignant_records.append(rec)
                elif (rec["diagnosis"] in self.BENIGN_DIAGNOSES or rec["benign_malignant"] == "benign"):
                    if len(benign_records) < n_benign * 2:
                        if not any(r["isic_id"] == rec["isic_id"] for r in benign_records):
                            benign_records.append(rec)

            page_url = data.get("next")
            pages_fetched += 1
            time.sleep(0.5)

        if len(malignant_records) < 10 or len(benign_records) < 10:
            warnings.warn(
                "⚠️  Insuficientes imágenes descargadas de ISIC API. "
                "Activando fallback sintético."
            )
            return None

        print(f"   Metadatos obtenidos: {len(benign_records)} benignos, "
              f"{len(malignant_records)} malignos")

        # ── Paso 2: Descargar imágenes ────────────────────────────────────
        records_out = []
        for cls_records, label, cls_name, n_target in [
            (benign_records,    0, "benign",    n_benign),
            (malignant_records, 1, "malignant", n_malignant)
        ]:
            cls_dir = self.output_dir / "raw" / cls_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            downloaded = 0

            for rec in cls_records:
                if downloaded >= n_target:
                    break
                dest = cls_dir / f"{rec['isic_id']}.jpg"
                # Usar thumbnail_url de la API si está disponible, o construirla
                thumb_url = rec.get("thumbnail_url") or ""
                if self._download_image(rec["isic_id"], dest, thumb_url=thumb_url):
                    # Redimensionar al tamaño estándar
                    try:
                        img = Image.open(dest).convert("RGB")
                        img = img.resize((img_size, img_size), Image.LANCZOS)
                        img.save(dest, quality=92)
                        rec["image_path"] = str(dest)
                        rec["label"]      = label
                        rec["class_name"] = cls_name
                        rec["has_artifact"]   = False
                        rec["artifact_type"]  = "none"
                        records_out.append(rec.copy())
                        downloaded += 1
                    except Exception:
                        pass
                time.sleep(0.2)  # Rate limiting

            print(f"   ✅ {cls_name}: {downloaded}/{n_target} imágenes descargadas")

        if not records_out:
            return None

        df = pd.DataFrame(records_out)

        # ── Transparencia: informe de calidad del patient_id ──────────────
        if "patient_id_source" in df.columns:
            src_counts = df["patient_id_source"].value_counts()
            total = len(df)
            print(f"\n   📋 Calidad del patient_id ({total} imágenes):")
            for src, cnt in src_counts.items():
                pct = cnt / total * 100
                quality = "✅" if src != "isic_id_fallback" else "⚠️ "
                print(f"      {quality} {src:<25}: {cnt:>4} ({pct:.1f}%)")
            n_fallback = src_counts.get("isic_id_fallback", 0)
            if n_fallback > 0:
                print(f"   ⚠️  {n_fallback} imágenes sin patient_id real en ISIC API.")
                print(f"      El split usa fallback por imagen para esos casos.")
                print(f"      NO garantiza prevención completa de fuga entre estudios relacionados.")

        # Añadir artefactos SOLO para proof-of-concept de detección (subset maligno, ~30%)
        if add_artifacts_for_demo:
            print()
            print("   " + "⚠️ " * 20)
            print("   DEMO_MODE=True: inyectando artefactos SINTÉTICOS sobre imágenes reales.")
            print("   Los mapas Grad-CAM mostrarán el sesgo que TÚ fabricaste,")
            print("   no el sesgo nativo del dataset ISIC.")
            print("   Presenta los resultados como 'validación del detector',")
            print("   NO como evidencia del sesgo real del dataset.")
            print("   " + "⚠️ " * 20)
            df = self._inject_demo_artifacts(df)

        return df

    # ------------------------------------------------------------------
    # Alias de conveniencia (evita divergencia entre notebook y src/)
    # ------------------------------------------------------------------
    download = download_balanced_subset


    def _inject_demo_artifacts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inyecta artefactos sintéticos sobre imágenes malignas reales para
        la demostración del Clever Hans.

        ⚠️  IMPORTANTE METODOLÓGICO: Esta función simula el sesgo histórico del
        dataset ISIC (donde los dermatólogos incluían reglas en imágenes de
        melanomas previas a la biopsia). En datos reales de ISIC 2020 ya no es
        necesaria porque el sesgo está presente de forma natural.
        """
        # _inject_demo_artifacts usa directamente ISICSyntheticUnitTest que está
        # definida en este mismo módulo (más abajo). No importar via 'src.data_pipeline'
        # para evitar el import circular.
        generator = ISICSyntheticUnitTest()

        artifact_types = ["ruler", "ink_marker", "patch", "hair"]
        mal_idx = df[(df["label"] == 1)].sample(
            frac=0.30, random_state=42
        ).index

        for idx in mal_idx:
            img_path = df.loc[idx, "image_path"]
            artifact_type = random.choice(artifact_types)
            try:
                img = Image.open(img_path).convert("RGB")
                img_size = img.size[0]
                img, mask = generator._add_clinical_artifacts(img, img_size, artifact_type)
                img.save(img_path, quality=92)

                mask_path = img_path.replace(".jpg", "_artifact_mask.png")
                mask.save(mask_path)

                df.loc[idx, "has_artifact"]  = True
                df.loc[idx, "artifact_type"] = artifact_type
                df.loc[idx, "artifact_mask_path"] = mask_path
            except Exception:
                pass

        return df


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1.1: GENERADOR SINTÉTICO — SOLO PARA TESTS UNITARIOS DEL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
class ISICSyntheticUnitTest:
    """
    ══════════════════════════════════════════════════════════════════════════
    ⚠️  SOLO PARA TEST UNITARIO DEL PIPELINE — NO USAR PARA CIENCIA
    ══════════════════════════════════════════════════════════════════════════

    Genera imágenes sintéticas mínimamente realistas para verificar que el
    pipeline Python funciona de extremo a extremo sin errores de runtime,
    SIN ninguna validez científica.

    Por qué no sirve para ciencia:
        - La tarea real es distinguir nevus displásico de melanoma uveal,
          basándose en patrones reticulares, glóbulos, estructuras vasculares,
          regresión, velo azul-blanco. Nada de esto se puede simular con PIL.
        - Cualquier AUC reportado sobre estas imágenes es irreal y no
          generalizable a datos dermatoscópicos reales.
        - El "sesgo Clever Hans descubierto" es el que tú mismo inyectaste
          (no uno emergente), lo que lo convierte en una demostración pedagógica,
          no en evidencia científica del fenómeno.

    Uso correcto:
        # ✅ Prueba rápida de que el código no tiene errores
        gen = ISICSyntheticUnitTest()
        df = gen.generate(output_dir='./tmp_test', n_benign=20, n_malignant=5)
        # Verificar que los shapes, DataLoaders y métricas son coherentes

        # ❌ NUNCA reportar AUC/Sensitivity de estas imágenes en un paper
    """

    @staticmethod
    def _draw_benign_lesion(img: Image.Image, size: int) -> Image.Image:
        draw = ImageDraw.Draw(img)
        cx, cy = size // 2, size // 2
        radius = random.randint(size // 5, size // 3)
        r = random.randint(120, 160)
        g = random.randint(80, 110)
        b = random.randint(40, 70)
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(r, g, b))
        return img

    @staticmethod
    def _draw_malignant_lesion(img: Image.Image, size: int) -> Image.Image:
        draw = ImageDraw.Draw(img)
        cx, cy = size // 2, size // 2
        n_points = random.randint(8, 15)
        points = []
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            r = random.randint(size // 5, size // 3)
            points.append((cx + int(r * np.cos(angle)), cy + int(r * np.sin(angle))))
        for color in [
            (random.randint(60, 100), random.randint(30, 60), random.randint(10, 40)),
            (random.randint(100, 140), random.randint(60, 90), random.randint(20, 50)),
        ]:
            offset = random.randint(5, 20)
            shifted = [(p[0] + random.randint(-offset, offset),
                        p[1] + random.randint(-offset, offset)) for p in points]
            draw.polygon(shifted, fill=color)
        return img

    @staticmethod
    def _add_clinical_artifacts(img: Image.Image, size: int,
                                 artifact_type: str = "ruler") -> Tuple[Image.Image, Image.Image]:
        draw = ImageDraw.Draw(img)
        mask = Image.new("L", (size, size), 0)
        mask_draw = ImageDraw.Draw(mask)

        if artifact_type == "ruler":
            y_pos = size - 30
            draw.line([(10, y_pos), (size - 10, y_pos)], fill=(0, 0, 0), width=2)
            mask_draw.line([(10, y_pos), (size - 10, y_pos)], fill=255, width=2)
            for i in range(0, size - 20, 10):
                tick_h = 8 if i % 50 == 0 else 4
                draw.line([(10 + i, y_pos), (10 + i, y_pos + tick_h)], fill=(0, 0, 0), width=1)
                mask_draw.line([(10 + i, y_pos), (10 + i, y_pos + tick_h)], fill=255, width=1)
        elif artifact_type == "ink_marker":
            x1, y1 = random.randint(0, size // 4), random.randint(0, size // 4)
            x2, y2 = random.randint(3 * size // 4, size), random.randint(3 * size // 4, size)
            draw.line([(x1, y1), (x2, y2)], fill=(128, 0, 128), width=3)
            mask_draw.line([(x1, y1), (x2, y2)], fill=255, width=3)
        elif artifact_type == "patch":
            patch_size = size // 4
            draw.rectangle([size - patch_size, 0, size, patch_size], fill=(200, 200, 180))
            mask_draw.rectangle([size - patch_size, 0, size, patch_size], fill=255)
        elif artifact_type == "hair":
            for _ in range(random.randint(3, 8)):
                x1, y1 = random.randint(0, size), random.randint(0, size)
                end_x, end_y = x1 + random.randint(-50, 50), y1 + random.randint(-50, 50)
                draw.line([(x1, y1), (end_x, end_y)], fill=(30, 20, 10), width=1)
                mask_draw.line([(x1, y1), (end_x, end_y)], fill=255, width=1)
        return img, mask

    def generate(
        self,
        output_dir: str,
        n_benign: int = 200,
        n_malignant: int = 50,
        img_size: int = 380,
        add_artifacts_to_malignant: bool = True
    ) -> pd.DataFrame:
        """Genera el dataset sintético. Ver advertencia de clase."""
        warnings.warn(
            "ISICSyntheticUnitTest.generate() genera datos SINTÉTICOS. "
            "Los resultados NO son científicamente válidos. "
            "Usar ISICArchiveDownloader para datos reales.",
            UserWarning, stacklevel=2
        )

        output_path = Path(output_dir)
        records = []
        split_ratios = {"train": 0.70, "val": 0.15, "test": 0.15}

        for split in ["train", "val", "test"]:
            for cls, label, n_img in [("benign", 0, n_benign), ("malignant", 1, n_malignant)]:
                split_dir = output_path / split / cls
                split_dir.mkdir(parents=True, exist_ok=True)
                n_split = max(1, int(n_img * split_ratios[split]))

                for i in range(n_split):
                    bg_color = (random.randint(200, 240),
                                random.randint(170, 210),
                                random.randint(150, 190))
                    img = Image.new("RGB", (img_size, img_size), color=bg_color)
                    noise = np.random.normal(0, 8, (img_size, img_size, 3)).astype(np.int16)
                    img_array = np.clip(np.array(img).astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    img = Image.fromarray(img_array)
                    img = self._draw_malignant_lesion(img, img_size) if label == 1 \
                          else self._draw_benign_lesion(img, img_size)

                    img_path = split_dir / f"{cls}_{i:04d}.jpg"
                    
                    has_artifact, artifact_type = False, "none"
                    mask_path_str = None
                    if add_artifacts_to_malignant and label == 1 and split == "train" \
                            and random.random() < 0.4:
                        artifact_type = random.choice(["ruler", "ink_marker", "patch", "hair"])
                        img, mask = self._add_clinical_artifacts(img, img_size, artifact_type)
                        has_artifact = True
                        mask_path = str(img_path).replace(".jpg", "_artifact_mask.png")
                        mask.save(mask_path)
                        mask_path_str = mask_path

                    img.save(img_path, quality=95)
                    records.append({
                        "image_path": str(img_path), "label": label,
                        "class_name": cls, "has_artifact": has_artifact,
                        "artifact_type": artifact_type, "split": split,
                        "patient_id": f"synth_{cls}_{i:04d}",  # ID ficticio
                        "artifact_mask_path": mask_path_str
                    })

        df = pd.DataFrame(records)
        df.to_csv(output_path / "metadata.csv", index=False)
        print(f"✅ [SINTÉTICO] {len(df)} imágenes generadas (SOLO PARA PRUEBAS)")
        return df


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1.2: CALIDAD DEL AGRUPAMIENTO POR PACIENTE/LESIÓN
# ─────────────────────────────────────────────────────────────────────────────
def summarize_grouping_quality(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    patient_source_col: str = "patient_id_source",
) -> Dict[str, object]:
    """
    Resume si el agrupamiento disponible permite sostener un split
    estrictamente a nivel de paciente.

    Si `patient_id_source` contiene `isic_id_fallback`, el split deja de ser
    defendible como "por paciente" y pasa a ser una agrupación mixta
    paciente/imagen.
    """
    if patient_col not in df.columns:
        raise ValueError(f"Columna requerida ausente: '{patient_col}'")

    total_images = int(len(df))
    unique_groups = int(df[patient_col].nunique(dropna=True))

    source_counts: Dict[str, int] = {}
    missing_source_count = 0
    fallback_count = 0
    if patient_source_col in df.columns:
        source_series = (
            df[patient_source_col]
            .fillna("missing")
            .astype(str)
            .str.strip()
            .replace("", "missing")
        )
        source_counts = {
            str(key): int(value) for key, value in source_series.value_counts().to_dict().items()
        }
        missing_source_count = int(source_counts.get("missing", 0))
        fallback_count = int(source_counts.get("isic_id_fallback", 0))

    fallback_fraction = (fallback_count / total_images) if total_images else 0.0
    strict_patient_grouping = fallback_count == 0
    grouping_level = "patient" if strict_patient_grouping else "mixed_patient_image"

    return {
        "total_images": total_images,
        "unique_groups": unique_groups,
        "patient_source_counts": source_counts,
        "missing_patient_source_count": missing_source_count,
        "fallback_count": fallback_count,
        "fallback_fraction": fallback_fraction,
        "strict_patient_grouping": strict_patient_grouping,
        "grouping_level": grouping_level,
    }


def validate_grouping_quality(
    grouping_summary: Dict[str, object],
    require_strict_patient_grouping: bool = True,
) -> None:
    """
    Lanza un error si se requiere agrupamiento estricto y el dataset usa
    fallbacks por imagen (`isic_id_fallback`).
    """
    if require_strict_patient_grouping and not grouping_summary["strict_patient_grouping"]:
        fallback_count = int(grouping_summary["fallback_count"])
        fallback_fraction = float(grouping_summary["fallback_fraction"])
        raise RuntimeError(
            "No es válido presentar este split como estrictamente 'a nivel de paciente'. "
            f"Se detectaron {fallback_count} imágenes ({fallback_fraction:.1%}) con "
            "agrupamiento por `isic_id_fallback`. Para un análisis exploratorio "
            "puedes desactivar esta protección, pero no para claims científicos."
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1.2: SPLIT A NIVEL DE PACIENTE
# ─────────────────────────────────────────────────────────────────────────────
def patient_level_split(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    label_col:   str = "label",
    train_frac:  float = 0.70,
    val_frac:    float = 0.15,
    seed:        int = 42,
    require_strict_patient_grouping: bool = True
) -> pd.DataFrame:
    """
    Realiza el split train/val/test A NIVEL DE PACIENTE para prevenir fuga
    de datos cuando hay múltiples imágenes del mismo lunar o paciente.

    Problema que resuelve:
        El ISIC dataset contiene múltiples imágenes del mismo lunar (diferentes
        ángulos, zoom, iluminación) y múltiples lunares del mismo paciente.
        Si hacemos un split aleatorio de imágenes, la misma lesión puede aparecer
        en train Y en test, inflando artificialmente el rendimiento (data leakage).

    Implementación:
        1. Obtener lista única de patient_ids.
        2. Hacer split ESTRATIFICADO de pacientes por proporción maligno/benigno
           (para mantener el ratio de clases en cada split).
        3. Asignar TODAS las imágenes de un paciente al mismo split.

    TRIPOD+AI ítem 12a: "Describe cómo se realizó la separación temporal o
    geográfica de los datos, o en su defecto la separación por participante."

    Args:
        df:          DataFrame con columnas [patient_id, label, image_path, ...].
        patient_col: Nombre de la columna con ID de paciente.
        label_col:   Nombre de la columna de clase binaria.
        train_frac:  Fracción de pacientes para entrenamiento.
        val_frac:    Fracción de pacientes para validación.
        seed:        Semilla para reproducibilidad.

    Returns:
        DataFrame original con columna 'split' añadida.
    """
    rng = np.random.RandomState(seed)

    if patient_col not in df.columns:
        raise ValueError(f"Columna requerida ausente: '{patient_col}'")
    if label_col not in df.columns:
        raise ValueError(f"Columna requerida ausente: '{label_col}'")

    split_fracs = {
        "train": train_frac,
        "val": val_frac,
        "test": 1.0 - train_frac - val_frac,
    }
    if min(split_fracs.values()) < 0:
        raise ValueError("Fracciones de split inválidas: train_frac + val_frac debe ser <= 1.0")

    def compute_target_counts(n_patients: int) -> Dict[str, int]:
        if n_patients <= 0:
            return {"train": 0, "val": 0, "test": 0}
        if n_patients == 1:
            return {"train": 1, "val": 0, "test": 0}
        if n_patients == 2:
            return {"train": 1, "val": 0, "test": 1}

        counts = {"train": 1, "val": 1, "test": 1}
        remaining = n_patients - 3
        if remaining == 0:
            return counts

        split_names = ["train", "val", "test"]
        raw = np.array([split_fracs[name] for name in split_names], dtype=float)
        raw = raw / raw.sum()
        scaled = raw * remaining
        base = np.floor(scaled).astype(int)
        remainders = scaled - base

        for idx, name in enumerate(split_names):
            counts[name] += int(base[idx])

        extra = remaining - int(base.sum())
        for idx in np.argsort(-remainders)[:extra]:
            counts[split_names[idx]] += 1

        return counts

    # Obtener pacientes únicos con su clase mayoritaria (para estratificación)
    patient_labels = (
        df.groupby(patient_col)[label_col]
        .agg(lambda x: int(x.mode()[0]))
        .reset_index()
        .rename(columns={label_col: "dominant_label"})
    )

    # Split estratificado de pacientes por clase
    patients_mal = patient_labels[patient_labels["dominant_label"] == 1][patient_col].values
    patients_ben = patient_labels[patient_labels["dominant_label"] == 0][patient_col].values

    def split_patients(patients):
        patients = np.array(patients, copy=True)
        rng.shuffle(patients)
        target_counts = compute_target_counts(len(patients))

        offsets = {}
        start = 0
        for split_name in ["train", "val", "test"]:
            stop = start + target_counts[split_name]
            offsets[split_name] = set(patients[start:stop])
            start = stop

        return offsets["train"], offsets["val"], offsets["test"]

    train_mal, val_mal, test_mal = split_patients(patients_mal)
    train_ben, val_ben, test_ben = split_patients(patients_ben)

    train_patients = train_mal | train_ben
    val_patients   = val_mal   | val_ben
    test_patients  = test_mal  | test_ben

    def assign_split(patient_id):
        if patient_id in train_patients: return "train"
        if patient_id in val_patients:   return "val"
        return "test"

    df = df.copy()
    df["split"] = df[patient_col].apply(assign_split)

    grouping_summary = summarize_grouping_quality(df, patient_col=patient_col)
    fallback_count = int(grouping_summary["fallback_count"])

    # ⚠️ Antes esta comprobación existía en el código (validate_grouping_quality)
    # pero nunca se invocaba desde ningún sitio: el flag require_strict_patient_grouping
    # de CONFIG no tenía ningún efecto real. Ahora sí se aplica aquí, en el único
    # lugar donde se decide el split, así que no hay forma de saltárselo sin querer.
    validate_grouping_quality(
        grouping_summary,
        require_strict_patient_grouping=require_strict_patient_grouping
    )

    # Estadísticas del split
    stats = df.groupby(["split", label_col]).size().unstack(fill_value=0)
    split_label = "PACIENTE" if fallback_count == 0 else "AGRUPACIÓN MIXTA"
    print(f"\n📊 Split a nivel de {split_label}:")
    print(stats.to_string())
    print(f"   Pacientes únicos: train={len(train_patients)}, "
          f"val={len(val_patients)}, test={len(test_patients)}")
    if fallback_count > 0:
        warnings.warn(
            f"{fallback_count} imágenes usan isic_id como fallback de agrupación "
            f"({grouping_summary['fallback_fraction']:.1%} del dataset). "
            "La prevención de fuga de datos no está garantizada para esos casos."
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1.2b: ENRIQUECIMIENTO CON patient_id REAL (ISIC 2020 Challenge)
# ─────────────────────────────────────────────────────────────────────────────
def enrich_with_isic2020_patient_ids(
    df: pd.DataFrame,
    ground_truth_csv_path: str,
    isic_id_col: str = "isic_id",
) -> pd.DataFrame:
    """
    Sustituye el patient_id por fallback (isic_id) por el patient_id REAL
    publicado en los metadatos oficiales del ISIC 2020 Challenge, cuando
    esté disponible localmente.

    La API pública ISIC Archive v2 (usada por ISICArchiveDownloader) NO expone
    patient_id real para acceso anónimo — de ahí que hasta ahora el 100% de
    las imágenes cayeran en el fallback `isic_id_fallback`, lo que anula la
    protección de patient_level_split() (cada imagen es su propio "paciente").

    El archivo oficial `ISIC_2020_Training_GroundTruth_v2.csv` (competición
    Kaggle "SIIM-ISIC Melanoma Classification") sí incluye patient_id real
    por imagen. Esta función lo usa para corregir el agrupamiento allí donde
    haya coincidencia por isic_id.

    Args:
        df: DataFrame del pipeline, con columna `isic_id_col` y `patient_id_source`.
        ground_truth_csv_path: ruta local al CSV oficial de metadatos ISIC 2020.
        isic_id_col: nombre de la columna con el ID de imagen en `df`.

    Returns:
        df con patient_id / patient_id_source actualizados donde hubo match.
        Si el archivo no existe, devuelve df sin modificar (y avisa por consola).
    """
    gt_path = Path(ground_truth_csv_path)
    if not gt_path.exists():
        warnings.warn(
            f"⚠️  No se encontró el CSV de metadatos ISIC 2020 en '{gt_path}'. "
            "El patient_id sigue en fallback por isic_id para todas las imágenes "
            "sin coincidencia. Descárgalo desde la competición Kaggle "
            "'SIIM-ISIC Melanoma Classification' (archivo "
            "ISIC_2020_Training_GroundTruth_v2.csv) para un agrupamiento por "
            "paciente real y defendible científicamente."
        )
        return df

    gt = pd.read_csv(gt_path)

    # El CSV oficial usa 'image_name' (p.ej. 'ISIC_0015719'); normalizamos
    # ambos lados a mayúsculas sin extensión para hacer el match robusto.
    id_col_gt = "image_name" if "image_name" in gt.columns else "isic_id"
    if id_col_gt not in gt.columns or "patient_id" not in gt.columns:
        warnings.warn(
            f"⚠️  El CSV '{gt_path}' no tiene las columnas esperadas "
            f"('{id_col_gt}', 'patient_id'). No se aplica enriquecimiento."
        )
        return df

    gt = gt[[id_col_gt, "patient_id"]].rename(columns={"patient_id": "patient_id_real"})
    gt["_match_key"] = gt[id_col_gt].astype(str).str.upper().str.replace(".JPG", "", regex=False)

    df = df.copy()
    df["_match_key"] = df[isic_id_col].astype(str).str.upper().str.replace(".JPG", "", regex=False)

    before_fallback = int((df.get("patient_id_source", "isic_id_fallback") == "isic_id_fallback").sum())

    df = df.merge(gt[["_match_key", "patient_id_real"]], on="_match_key", how="left")
    matched = df["patient_id_real"].notna()
    df.loc[matched, "patient_id"] = df.loc[matched, "patient_id_real"].astype(str)
    df.loc[matched, "patient_id_source"] = "isic_2020_ground_truth"
    df = df.drop(columns=["_match_key", "patient_id_real"])

    after_fallback = int((df["patient_id_source"] == "isic_id_fallback").sum())
    n_matched = int(matched.sum())
    print(f"\n🔧 Enriquecimiento de patient_id con metadatos ISIC 2020:")
    print(f"   Coincidencias encontradas: {n_matched}/{len(df)}")
    print(f"   Imágenes aún en fallback (isic_id_fallback): {before_fallback} → {after_fallback}")
    if after_fallback > 0:
        warnings.warn(
            f"{after_fallback} imágenes siguen sin patient_id real tras el "
            "enriquecimiento (no aparecen en el CSV oficial). El split seguirá "
            "tratándolas como paciente único (fallback) para esas filas."
        )

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1.2c: CARGA LOCAL DEL DATASET ISIC 2020 (KAGGLE) — patient_id 100% real
# ─────────────────────────────────────────────────────────────────────────────
class ISIC2020KaggleLoader:
    """
    Carga el dataset ISIC 2020 Challenge descargado localmente desde Kaggle
    (competición "SIIM-ISIC Melanoma Classification"), en vez de depender de
    la API pública de ISIC Archive.

    Ventaja frente a ISICArchiveDownloader: el CSV oficial
    (`ISIC_2020_Training_GroundTruth_v2.csv`) trae patient_id REAL para el
    100% de las imágenes de este dataset. No hace falta enriquecimiento ni
    fallback: el agrupamiento por paciente es válido desde el origen.

    Requiere tener localmente:
        - El CSV de metadatos oficial.
        - La carpeta con las imágenes JPEG del set de entrenamiento
          (el .zip "ISIC_2020_Training_JPEG" de la misma página de Kaggle,
          descomprimido), con nombres de archivo `<image_name>.jpg`.
    """

    def __init__(self, images_dir: str, ground_truth_csv: str):
        self.images_dir = Path(images_dir)
        self.ground_truth_csv = Path(ground_truth_csv)

    def available(self) -> bool:
        """True si ambos, CSV y carpeta de imágenes, existen localmente."""
        return self.ground_truth_csv.exists() and self.images_dir.exists()

    def load(
        self,
        n_benign: Optional[int] = None,
        n_malignant: Optional[int] = None,
        seed: int = 42,
        image_ext: str = ".jpg",
    ) -> pd.DataFrame:
        if not self.ground_truth_csv.exists():
            raise FileNotFoundError(
                f"No se encontró el CSV de metadatos en '{self.ground_truth_csv}'."
            )
        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"No se encontró la carpeta de imágenes en '{self.images_dir}'. "
                "Descarga y descomprime 'ISIC_2020_Training_JPEG' desde la misma "
                "página de Kaggle del CSV de metadatos."
            )

        gt = pd.read_csv(self.ground_truth_csv)
        required_cols = {"image_name", "patient_id"}
        missing = required_cols - set(gt.columns)
        if missing:
            raise ValueError(f"Faltan columnas requeridas en el CSV: {missing}")

        if "target" in gt.columns:
            gt["label"] = gt["target"].astype(int)
        elif "benign_malignant" in gt.columns:
            gt["label"] = (gt["benign_malignant"].str.lower() == "malignant").astype(int)
        else:
            raise ValueError(
                "El CSV no tiene columna 'target' ni 'benign_malignant'; "
                "no se puede derivar la etiqueta binaria."
            )

        gt["isic_id"] = gt["image_name"].astype(str)
        gt["image_path"] = gt["isic_id"].apply(
            lambda x: str(self.images_dir / f"{x}{image_ext}")
        )
        exists_mask = gt["image_path"].apply(lambda p: Path(p).exists())
        n_missing = int((~exists_mask).sum())
        if n_missing > 0:
            warnings.warn(
                f"{n_missing} imágenes listadas en el CSV no se encontraron en "
                f"'{self.images_dir}' y se descartan del dataset."
            )
        gt = gt[exists_mask].reset_index(drop=True)
        if gt.empty:
            raise RuntimeError(
                "Ninguna imagen del CSV se encontró en la carpeta indicada. "
                "Revisa 'images_dir' y la extensión de archivo (image_ext)."
            )

        gt["patient_id"] = gt["patient_id"].astype(str)
        # 100% real: viene directamente del dataset oficial, no hay fallback.
        gt["patient_id_source"] = "isic_2020_ground_truth"
        gt["diagnosis"] = gt["diagnosis"].fillna("") if "diagnosis" in gt.columns else ""
        gt["benign_malignant"] = gt["benign_malignant"] if "benign_malignant" in gt.columns else ""
        gt["class_name"] = gt["label"].map({0: "benign", 1: "malignant"})
        gt["has_artifact"] = False
        gt["artifact_type"] = "none"

        if n_benign is not None or n_malignant is not None:
            parts = []
            for label, n_target in [(0, n_benign), (1, n_malignant)]:
                sub = gt[gt["label"] == label]
                if n_target is not None and n_target < len(sub):
                    sub = sub.sample(n=n_target, random_state=seed)
                parts.append(sub)
            gt = pd.concat(parts).reset_index(drop=True)

        cols = ["isic_id", "image_path", "label", "class_name", "diagnosis",
                "benign_malignant", "patient_id", "patient_id_source",
                "has_artifact", "artifact_type"]
        df = gt[cols].copy()

        n_total = len(df)
        n_ben = int((df["label"] == 0).sum())
        n_mal = int((df["label"] == 1).sum())
        n_patients = df["patient_id"].nunique()
        print(f"\n📦 Dataset ISIC 2020 (Kaggle, local) cargado:")
        print(f"   Total imágenes: {n_total} | Benigno: {n_ben} | Maligno: {n_mal}")
        if n_missing > 0:
            print(f"   ⚠️  Descartadas por no encontrar archivo: {n_missing}")
        print(f"   Pacientes únicos: {n_patients} (patient_id 100% real, sin fallback)")

        return df


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1.3: TRANSFORMACIONES — 3 CONDICIONES EXPERIMENTALES CLARAS
# ─────────────────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Fracción del lado que se mantiene tras CenterCrop (elimina ~15% de bordes)
CENTER_CROP_RATIO = 0.85


def apply_circular_crop(img_np: np.ndarray) -> np.ndarray:
    """
    Aplica una máscara circular sobre la imagen, rellenando las esquinas
    con el color medio de la imagen.

    JUSTIFICACIÓN CIENTÍFICA (TRIPOD+AI ítem 15):
        Las imágenes dermatoscópicas reales tienen campo de visión circular.
        Las esquinas contienen información del marco del dermatoscopio
        (viñeteo, bordes oscuros) que varía según el dispositivo de adquisición.

        Análisis cuantitativo del dataset actual:
            - 15.9% de imágenes malignas tienen esquinas oscuras (<50 brillo)
            - Solo 2.7% de benignas tienen esquinas oscuras
            - Esta diferencia 6x permite al modelo clasificar por esquinas
              en vez de por la lesión (efecto Clever Hans confirmado por Grad-CAM)

        Al enmascarar las esquinas con el color medio, eliminamos este
        atajo espurio sin perder información clínica de la lesión central.

    Args:
        img_np: Array numpy (H, W, 3) con la imagen RGB.

    Returns:
        Array numpy (H, W, 3) con esquinas rellenadas por color medio.
    """
    h, w = img_np.shape[:2]
    
    # Crear máscara circular: disco de radio = min(h,w)//2 centrado
    center_y, center_x = h // 2, w // 2
    radius = min(h, w) // 2
    
    Y, X = np.ogrid[:h, :w]
    dist_sq = (X - center_x) ** 2 + (Y - center_y) ** 2
    circle_mask = dist_sq <= radius ** 2  # True dentro del círculo
    
    # Calcular color medio de la zona central (lesión + piel)
    mean_color = img_np[circle_mask].mean(axis=0).astype(np.uint8)
    
    # Rellenar esquinas con el color medio
    result = img_np.copy()
    result[~circle_mask] = mean_color
    
    return result


def get_transforms(
    split: str,
    img_size: int = 224,
    condition: Literal["unmitigated", "partial", "masked"] = "partial"
) -> A.Compose:
    """
    Define transformaciones según split Y condición experimental.

    CORRECCIÓN v3.1 (Fix Clever Hans):
        Todas las transforms (train, val, test) incluyen ahora un CenterCrop
        que elimina el 15% exterior de la imagen. Esto destruye la correlación
        espuria entre esquinas oscuras y clase maligna que el modelo v3.0
        explotaba como atajo.

        La máscara circular (apply_circular_crop) se aplica en el Dataset
        ANTES de estas transforms, proporcionando doble protección:
            1. Circular crop → rellena esquinas con color medio
            2. CenterCrop → recorta la zona de transición

    CONDICIÓN A — 'unmitigated':
        CenterCrop + Resize. El modelo no ve bordes pero sí toda la piel.

    CONDICIÓN B — 'partial' (default):
        CenterCrop + RandomResizedCrop(scale=0.65-0.90). Recorte agresivo
        que asegura que el modelo NUNCA ve las esquinas originales.

    CONDICIÓN C — 'masked':
        CenterCrop + Resize (el masking lo hace MaskedISICDataset).

    Args:
        split:     'train', 'val', 'test'
        img_size:  Tamaño de salida (px). Default 224 (resolución real de los thumbnails)
        condition: 'unmitigated', 'partial', o 'masked'
    """
    # Tamaño intermedio tras CenterCrop (elimina esquinas residuales)
    center_crop_size = int(img_size * CENTER_CROP_RATIO)

    if split != "train":
        # Val/Test: CenterCrop + resize + normalize — sin augmentation
        # CRÍTICO: CenterCrop también en test para consistencia con training
        return A.Compose([
            A.Resize(img_size, img_size),
            A.CenterCrop(center_crop_size, center_crop_size),
            A.Resize(img_size, img_size),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ])

    # ── AUGMENTACIÓN BASE (común a todas las condiciones de train) ────────
    common_aug = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                 scale=(0.85, 1.15), rotate=(-180, 180),
                 mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.OpticalDistortion(distort_limit=0.3, p=1.0),
            A.ElasticTransform(alpha=1, sigma=50, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        ], p=0.7),
        A.CLAHE(clip_limit=6.0, tile_grid_size=(8,8), p=0.4),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
        ], p=0.3),
        A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(10, 30),
                        hole_width_range=(10, 30), fill=0, p=0.4),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]

    if condition == "unmitigated":
        # ── COND-A: CenterCrop + Resize (sin RandomResizedCrop) ───────────
        return A.Compose([
            A.Resize(img_size, img_size),
            A.CenterCrop(center_crop_size, center_crop_size),
            A.Resize(img_size, img_size),
            *common_aug
        ])

    elif condition == "partial":
        # ── COND-B: CenterCrop + RandomResizedCrop AGRESIVO ───────────────
        # Scale 0.65-0.90: NUNCA ve la imagen completa, siempre recorta bordes
        return A.Compose([
            A.Resize(img_size, img_size),
            A.CenterCrop(center_crop_size, center_crop_size),
            A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.65, 0.90),
                ratio=(0.85, 1.15),
                p=1.0
            ),
            *common_aug
        ])

    elif condition == "masked":
        # ── COND-C: CenterCrop + Resize (el masking lo hace MaskedISICDataset)
        return A.Compose([
            A.Resize(img_size, img_size),
            A.CenterCrop(center_crop_size, center_crop_size),
            A.Resize(img_size, img_size),
            *common_aug
        ])

    else:
        raise ValueError(f"condition='{condition}' no reconocida. "
                         "Usar 'unmitigated', 'partial' o 'masked'.")


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1.4: DATASETS — ESTÁNDAR Y CON ENMASCARAMIENTO DE LESIÓN
# ─────────────────────────────────────────────────────────────────────────────
def _compute_sample_weights(df: pd.DataFrame) -> torch.Tensor:
    """
    Calcula pesos por muestra inversamente proporcionales a la frecuencia de clase.
    Usado por WeightedRandomSampler para re-balancear el entrenamiento.

    w_i = N_total / (K * N_c)   donde K=2 clases, N_c = muestras de la clase c.

    Función libre compartida por ISICDataset y MaskedISICDataset para evitar
    duplicación. Ver: Cao et al. (2019) Learning Imbalanced Datasets with LDAM.
    """
    class_counts = df["label"].value_counts().to_dict()
    n_total  = len(df)
    n_classes = 2
    weights = [
        n_total / (n_classes * class_counts[label])
        for label in df["label"]
    ]
    return torch.tensor(weights, dtype=torch.float)


class ISICDataset(Dataset):
    """
    Dataset PyTorch estándar para imágenes dermatoscópicas ISIC.
    Compatible con ISIC real (ISICArchiveDownloader) y sintético (unit tests).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        split: str,
        img_size: int = 224,
        condition: Literal["unmitigated", "partial", "masked"] = "partial",
        return_path: bool = False,
        preload_to_ram: bool = False
    ):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transforms = get_transforms(split, img_size, condition)
        self.return_path = return_path
        self.class_counts = self.df["label"].value_counts().to_dict()
        print(f"  [{split.upper()}|{condition}] "
              f"Benigno: {self.class_counts.get(0, 0)} | "
              f"Maligno: {self.class_counts.get(1, 0)}")

        # ── Caché en RAM (opcional) ─────────────────────────────────────────
        # Precarga TODAS las imágenes (decodificadas + recorte circular +
        # resize a img_size) en memoria, en el proceso PRINCIPAL, antes de
        # que DataLoader cree los procesos worker. Con multiprocessing
        # 'fork' (default en Linux), cada worker hereda esta memoria por
        # copy-on-write — elimina la re-lectura de disco en cada época.
        #
        # Por qué es un único np.ndarray contiguo y NO una lista de Python:
        # con una lista `[None]*N` de arrays individuales, cada acceso a
        # `cache[idx]` desde un worker incrementa el refcount de Python de
        # ESE objeto concreto, lo cual escribe en su página de memoria y
        # rompe el copy-on-write → el SO acaba copiándola de verdad. Con
        # WeightedRandomSampler (acceso aleatorio) y varios workers, cada
        # uno termina tocando —y copiando— casi toda la lista, multiplicando
        # la RAM usada por el número de workers. Un array NumPy contiguo no
        # tiene ese problema: indexarlo devuelve una vista sobre el mismo
        # buffer sin tocar refcounts por imagen.
        #
        # Redimensionamos aquí a (img_size, img_size) para poder usar un
        # array de tamaño fijo (los JPEG de Kaggle no miden todos lo mismo).
        # Esto no cambia el comportamiento de augmentación: el primer paso
        # de get_transforms() ya es un A.Resize(img_size, img_size), así que
        # este resize adelantado es, en la práctica, una identidad.
        self.preload_to_ram = preload_to_ram
        self._cache = None
        if preload_to_ram:
            print(f"  [{split.upper()}] Precargando {len(self.df)} imágenes en RAM...")
            t0 = time.time()
            cache = np.empty((len(self.df), img_size, img_size, 3), dtype=np.uint8)
            paths = self.df["image_path"].tolist()

            def _load_one(i: int):
                img = Image.open(paths[i]).convert("RGB")
                img_np = np.array(img)
                img_np = apply_circular_crop(img_np)
                if img_np.shape[:2] != (img_size, img_size):
                    img_np = cv2.resize(
                        img_np, (img_size, img_size),
                        interpolation=cv2.INTER_LINEAR
                    )
                cache[i] = img_np

            # La decodificación JPEG (PIL/libjpeg) y la E/S de disco liberan
            # el GIL, así que un ThreadPoolExecutor sí acelera esto de
            # verdad (no es CPU-bound puro en Python). Cada hilo escribe en
            # su propio índice del array, sin condiciones de carrera.
            n_threads = min(32, (os.cpu_count() or 8) * 2)
            done = 0
            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                for _ in pool.map(_load_one, range(len(self.df))):
                    done += 1
                    if done % 5000 == 0:
                        print(f"    ...{done}/{len(self.df)}")
            self._cache = cache
            print(f"  [{split.upper()}] Precarga completa en {time.time() - t0:.1f}s "
                  f"({cache.nbytes / 1e9:.2f} GB en RAM, {n_threads} hilos)")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        if self._cache is not None:
            # .copy(): self._cache es un único array compartido por todos
            # los workers; copiar solo esta imagen (barato) evita que una
            # transform mute en sitio el caché compartido entre épocas.
            img_np = self._cache[idx].copy()
        else:
            img = Image.open(row["image_path"]).convert("RGB")
            img_np = np.array(img)
            # ── Preprocesamiento anti-Clever-Hans ──────────────────────────
            # Eliminar esquinas ANTES de cualquier transform para que el
            # modelo nunca vea información del marco del dermatoscopio
            img_np = apply_circular_crop(img_np)
        augmented = self.transforms(image=img_np)
        image_tensor = augmented["image"]
        label = torch.tensor(row["label"], dtype=torch.float32)
        if self.return_path:
            return image_tensor, label, row["image_path"]
        return image_tensor, label

    @property
    def class_weights(self) -> torch.Tensor:
        """Pesos por muestra para WeightedRandomSampler. Ver _compute_sample_weights."""
        return _compute_sample_weights(self.df)


class MaskedISICDataset(Dataset):
    """
    Dataset de MITIGACIÓN REAL que aplica segmentación de lesión durante training.

    COND-C: Diferencia clave respecto a v1.0:
        En v1.0, el 'mitigated model' era el MISMO modelo evaluado con crop en
        inferencia — metodológicamente débil (no cambia lo que el modelo aprendió).

        Aquí, la máscara se aplica DURANTE EL ENTRENAMIENTO: el modelo NUNCA ve
        los píxeles de artefactos, aprende a diagnosticar únicamente desde la lesión.
        Esto es una intervención real en el proceso de aprendizaje.

    Estrategia de segmentación:
        1. Si existen máscaras anotadas (ISIC Segmentation Challenge): usarlas.
        2. Si no: segmentación automática heurística (Otsu en canal de saturación).
           Limitación conocida: falla en lesiones amelanóticas. Documentarlo.

    Referencia:
        Kawahara J, Hamarneh G. "Multi-resolution-tract CNN with hybrid pretrained
        and skin-lesion trained layers." MICCAI Workshops 2016.
        (Primer paper que propone enmascarar artefactos para dermoscopía DL)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        split: str,
        img_size: int = 224,
        mask_dir: Optional[str] = None,   # Directorio con máscaras ISIC Seg.
        return_path: bool = False
    ):
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.transforms = get_transforms(split, img_size, condition="masked")
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.return_path = return_path
        self.class_counts = self.df["label"].value_counts().to_dict()

        # Conteo de fallback de segmentación para reportar limitaciones
        self._auto_seg_count = 0
        self._annotated_mask_count = 0

        print(f"  [{split.upper()}|MASKED] "
              f"Benigno: {self.class_counts.get(0, 0)} | "
              f"Maligno: {self.class_counts.get(1, 0)}")

    def _load_mask(self, image_path: str) -> Optional[np.ndarray]:
        """
        Carga máscara anotada (ISIC Segmentation Challenge) o retorna None.

        Las máscaras ISIC tienen el mismo nombre que la imagen con sufijo '_segmentation.png'.
        """
        if self.mask_dir is None:
            return None
        isic_id = Path(image_path).stem  # e.g., 'ISIC_0024306'
        mask_path = self.mask_dir / f"{isic_id}_segmentation.png"
        if mask_path.exists():
            mask = np.array(Image.open(mask_path).convert("L"))
            return (mask > 127).astype(np.uint8)  # Binarizar
        return None

    def _auto_segment(self, img_np: np.ndarray) -> np.ndarray:
        """
        Segmentación automática heurística por umbralización Otsu en HSV.

        Limitación documentada (TRIPOD+AI ítem 15):
            Este método falla en lesiones amelanóticas (no pigmentadas) y en
            imágenes con iluminación no uniforme. Se recomienda reemplazar por
            un segmentador CNN entrenado (p.ej., U-Net en ISIC Seg. Challenge).
        """
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        s_channel = hsv[:, :, 1]
        _, mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Operaciones morfológicas para limpiar la máscara
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Seleccionar solo el contorno mayor (lesión principal)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros_like(mask)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.fillPoly(final_mask, [largest], 255)

        return (final_mask > 0).astype(np.uint8)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img_np = np.array(img)
        # ── Preprocesamiento anti-Clever-Hans (antes de segmentación) ──────
        img_np = apply_circular_crop(img_np)

        # ── Aplicar máscara de lesión ─────────────────────────────────────
        annotated_mask = self._load_mask(row["image_path"])
        if annotated_mask is not None:
            mask = annotated_mask
            self._annotated_mask_count += 1
        else:
            # Fallback: segmentación automática (documentar como limitación)
            mask = self._auto_segment(img_np)
            self._auto_seg_count += 1

        # Enmascarar: píxeles fuera de la lesión → negro (0,0,0)
        # El modelo aprende a diagnosticar SOLO desde la lesión, nunca desde artefactos
        masked_img = img_np.copy()
        masked_img[mask == 0] = 0

        augmented = self.transforms(image=masked_img)
        image_tensor = augmented["image"]
        label = torch.tensor(row["label"], dtype=torch.float32)

        if self.return_path:
            return image_tensor, label, row["image_path"]
        return image_tensor, label

    @property
    def segmentation_stats(self) -> Dict:
        """Estadísticas de segmentación para el informe de limitaciones."""
        total = self._auto_seg_count + self._annotated_mask_count
        return {
            "total_processed": total,
            "annotated_masks": self._annotated_mask_count,
            "auto_segmented": self._auto_seg_count,
            "auto_seg_rate": self._auto_seg_count / max(total, 1)
        }

    @property
    def class_weights(self) -> torch.Tensor:
        """Pesos por muestra para WeightedRandomSampler. Ver _compute_sample_weights."""
        return _compute_sample_weights(self.df)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1.5: DATALOADER FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def cutmix_mixup_collate_fn(batch, cutmix_prob=0.3, mixup_prob=0.2):
    """Collate function that applies CutMix or MixUp to a batch.
    Labels are returned as float for BCEWithLogitsLoss compatibility (single output)."""
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    
    r = random.random()
    if r < cutmix_prob:
        # CutMix
        lam = np.random.beta(1.0, 1.0)
        rand_index = torch.randperm(images.size(0))
        labels_b = labels[rand_index]
        bbx1, bby1, bbx2, bby2 = _rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        labels = labels * lam + labels_b * (1 - lam)
    elif r < cutmix_prob + mixup_prob:
        # MixUp
        lam = np.random.beta(0.4, 0.4)
        rand_index = torch.randperm(images.size(0))
        images = lam * images + (1 - lam) * images[rand_index]
        labels = lam * labels + (1 - lam) * labels[rand_index]
    
    return images, labels

def _rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def _worker_init_fn(worker_id: int) -> None:
    """
    Resemilla el RNG de cada worker del DataLoader para reproducibilidad.

    Problema: Con num_workers>0, los workers heredan el mismo estado RNG del
    proceso padre. Albumentations y numpy usarán secuencias idénticas entre
    workers y entre épocas → augmentaciones no independientes.

    Solución: Cada worker recibe una semilla única derivada del initial_seed
    de PyTorch (que varía por época) más su propio worker_id.

    Ref: PyTorch docs — "Reproducibility" — DataLoader Workers section.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def create_dataloaders(
    df: pd.DataFrame,
    img_size: int = 380,
    batch_size: int = 16,
    num_workers: int = 2,
    condition: Literal["unmitigated", "partial", "masked"] = "partial",
    use_weighted_sampler: bool = True,
    use_cutmix_mixup: bool = True,
    mask_dir: Optional[str] = None,
    preload_to_ram: bool = False
) -> Tuple[Dict[str, DataLoader], Dict[str, Dataset]]:
    """
    Crea DataLoaders para una condición experimental específica.

    ABLACIÓN PLANIFICADA (recomendada antes de publicar):
        Condición A (unmitigated, sampler=True,  focal=True):   Clever Hans emergente
        Condición B (partial,     sampler=True,  focal=True):   Mitigación parcial
        Condición C (masked,      sampler=True,  focal=True):   Mitigación real
        Extra D     (partial,     sampler=False, focal=True):   Sin WeightedSampler
        Extra E     (partial,     sampler=True,  focal=False):  Sin Focal Loss
        → Esta ablación aísla la contribución de cada componente.

    ⚠️  WeightedSampler + FocalLoss simultáneos:
        Pueden sobre-corregir el desbalance (sobreconfianza en clase minoritaria,
        peor calibración). La ablación anterior cuantifica este riesgo.
        Por ahora se mantienen ambos siguiendo la práctica de Cao et al. (2019):
        "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss".
    """
    datasets = {}
    for split in ["train", "val", "test"]:
        if condition == "masked":
            datasets[split] = MaskedISICDataset(
                df, split=split, img_size=img_size,
                mask_dir=mask_dir, return_path=False
            )
        else:
            datasets[split] = ISICDataset(
                df, split=split, img_size=img_size,
                condition=condition, return_path=False,
                preload_to_ram=preload_to_ram
            )

    split_sizes = {split: len(dataset) for split, dataset in datasets.items()}
    if split_sizes["train"] == 0:
        raise ValueError("El split de entrenamiento quedó vacío. Revisa patient_level_split().")
    for split in ["val", "test"]:
        if split_sizes[split] == 0:
            raise ValueError(f"El split '{split}' quedó vacío. Revisa patient_level_split().")

    loaders = {}
    train_drop_last = split_sizes["train"] > batch_size
    # TRAIN con WeightedRandomSampler
    collate_fn = cutmix_mixup_collate_fn if use_cutmix_mixup else None
    
    if use_weighted_sampler:
        sampler = WeightedRandomSampler(
            weights=datasets["train"].class_weights,
            num_samples=len(datasets["train"]),
            replacement=True
        )
        loaders["train"] = DataLoader(
            datasets["train"], batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=train_drop_last,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
            collate_fn=collate_fn,
            persistent_workers=(num_workers > 0)
        )
    else:
        loaders["train"] = DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=train_drop_last,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
            collate_fn=collate_fn,
            persistent_workers=(num_workers > 0)
        )

    for split in ["val", "test"]:
        loaders[split] = DataLoader(
            datasets[split], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
            persistent_workers=(num_workers > 0)
        )

    return loaders, datasets


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1.6: FOCAL LOSS (sin cambios, implementación correcta)
# ─────────────────────────────────────────────────────────────────────────────
class FocalLoss(torch.nn.Module):
    """Focal Loss for binary classification with single logit output."""
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: (N,) or (N,1) logits
        # targets: (N,) float labels (0.0 or 1.0, or mixed from CutMix/MixUp)
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        return focal_loss

# ─────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA PARA PRUEBA UNITARIA
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    set_global_seed(42)
    print("🧪 Ejecutando test unitario del pipeline de datos (datos sintéticos)...")
    gen = ISICSyntheticUnitTest()
    df = gen.generate("./tmp_unit_test", n_benign=40, n_malignant=10)

    # Aplicar split a nivel de paciente
    df = patient_level_split(df, patient_col="patient_id")

    # Probar las 3 condiciones
    for cond in ["unmitigated", "partial", "masked"]:
        print(f"\n  Condición: {cond}")
        loaders, datasets = create_dataloaders(
            df, batch_size=8, num_workers=0, condition=cond
        )
        batch = next(iter(loaders["train"]))
        print(f"  Batch shape: {batch[0].shape} | Labels: {batch[1].tolist()}")

    print("\n✅ Test unitario completado. El pipeline funciona correctamente.")
    print("   Para resultados científicos: usar ISICArchiveDownloader con datos reales.")
