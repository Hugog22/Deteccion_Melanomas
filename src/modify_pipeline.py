import re

with open('/Users/hugo/Documents/PROYECTOS/Detección_Melanomas/src/data_pipeline.py', 'r') as f:
    code = f.read()

# 1. Imports
code = re.sub(
    r'import torch\nfrom torch.utils.data',
    'from functools import partial\nimport torch\nimport torch.nn.functional as F\nfrom torch.utils.data',
    code
)

# 2. _get_json_filtered and _parse_img_meta in ISICArchiveDownloader
# We will just replace download_balanced_subset completely and add the new methods.
old_download_str = """    def download_balanced_subset(
        self,
        n_benign: int = 200,
        n_malignant: int = 100,
        img_size: int = 224,"""

new_download_str = """    def _get_json_filtered(self, diagnosis_filter: str) -> Optional[dict]:
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
        img_size: int = 380,"""

code = code.replace(old_download_str, new_download_str)

# Now replace the body of download_balanced_subset up to the download images step
start_marker = "        print(\"🌐 Intentando conectar con ISIC Archive API v2...\")"
end_marker = "        # ── Paso 2: Descargar imágenes ────────────────────────────────────"

new_fetch_logic = """        print("🌐 Intentando conectar con ISIC Archive API v2...")

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

"""

idx_start = code.find(start_marker)
idx_end = code.find(end_marker)
if idx_start != -1 and idx_end != -1:
    code = code[:idx_start] + new_fetch_logic + code[idx_end:]


# Update img_size defaults and other places
code = code.replace("img_size: int = 224", "img_size: int = 380")

# Update get_transforms
old_common_aug = """    # ── AUGMENTACIÓN BASE (común a todas las condiciones de train) ────────
    common_aug = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30,
                             val_shift_limit=20, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 16),
                        hole_width_range=(8, 16), fill=0, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]

    if condition == "unmitigated":
        # ── COND-A: Solo resize, sin RandomResizedCrop ────────────────────
        return A.Compose([
            A.Resize(img_size, img_size),  # Imagen completa visible (artefactos incluidos)
            *common_aug
        ])

    elif condition == "partial":
        # ── COND-B: Con RandomResizedCrop (mitigación parcial de bordes) ──
        return A.Compose([
            A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.80, 1.00),  # Elimina hasta 20% de bordes aleatoriamente
                ratio=(0.75, 1.33),
                p=1.0
            ),
            *common_aug
        ])"""

new_common_aug = """    # ── AUGMENTACIÓN BASE (común a todas las condiciones de train) ────────
    common_aug = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=180, 
                           border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
            A.OpticalDistortion(distort_limit=0.3, shift_limit=0.1, p=1.0),
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
        # ── COND-A: Solo resize, sin RandomResizedCrop ────────────────────
        return A.Compose([
            A.Resize(img_size, img_size),
            *common_aug
        ])

    elif condition == "partial":
        # ── COND-B: Con RandomResizedCrop (mitigación parcial de bordes) ──
        return A.Compose([
            A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.75, 1.00),
                ratio=(0.75, 1.33),
                p=1.0
            ),
            *common_aug
        ])"""

code = code.replace(old_common_aug, new_common_aug)

# Datasets Float Label
code = code.replace('label = torch.tensor(row["label"], dtype=torch.long)', 'label = torch.tensor(row["label"], dtype=torch.float32)')

# DataLoaders & CutMix/MixUp
collate_fn_code = """def cutmix_mixup_collate_fn(batch, cutmix_prob=0.3, mixup_prob=0.2):
    \"\"\"Collate function that applies CutMix or MixUp to a batch.
    Labels are returned as float for BCEWithLogitsLoss compatibility (single output).\"\"\"
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

def _worker_init_fn(worker_id: int) -> None:"""

code = code.replace("def _worker_init_fn(worker_id: int) -> None:", collate_fn_code)


old_create_dataloaders = """def create_dataloaders(
    df: pd.DataFrame,
    img_size: int = 380,
    batch_size: int = 32,
    num_workers: int = 2,
    condition: Literal["unmitigated", "partial", "masked"] = "partial",
    use_weighted_sampler: bool = True,
    mask_dir: Optional[str] = None
) -> Tuple[Dict[str, DataLoader], Dict[str, Dataset]]:"""

new_create_dataloaders = """def create_dataloaders(
    df: pd.DataFrame,
    img_size: int = 380,
    batch_size: int = 16,
    num_workers: int = 2,
    condition: Literal["unmitigated", "partial", "masked"] = "partial",
    use_weighted_sampler: bool = True,
    use_cutmix_mixup: bool = True,
    mask_dir: Optional[str] = None
) -> Tuple[Dict[str, DataLoader], Dict[str, Dataset]]:"""

code = code.replace(old_create_dataloaders, new_create_dataloaders)

old_train_loaders = """    # TRAIN con WeightedRandomSampler
    if use_weighted_sampler:
        sampler = WeightedRandomSampler(
            weights=datasets["train"].class_weights,
            num_samples=len(datasets["train"]),
            replacement=True
        )
        loaders["train"] = DataLoader(
            datasets["train"], batch_size=batch_size, sampler=sampler,
            num_workers=num_workers, pin_memory=True, drop_last=train_drop_last,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None
        )
    else:
        loaders["train"] = DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=train_drop_last,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None
        )"""

new_train_loaders = """    # TRAIN con WeightedRandomSampler
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
            collate_fn=collate_fn
        )
    else:
        loaders["train"] = DataLoader(
            datasets["train"], batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=train_drop_last,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
            collate_fn=collate_fn
        )"""

code = code.replace(old_train_loaders, new_train_loaders)

# FocalLoss update
focal_loss_start = "class FocalLoss(torch.nn.Module):"
# Need to replace until compute_focal_loss_alpha inclusive
import re
code = re.sub(r'class FocalLoss\(torch\.nn\.Module\):.*?(?=# ─────────────────────────────────────────────────────────────────────────────\n# PUNTO DE ENTRADA PARA PRUEBA UNITARIA)',
r'''class FocalLoss(torch.nn.Module):
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

''', code, flags=re.DOTALL)

with open('/Users/hugo/Documents/PROYECTOS/Detección_Melanomas/src/data_pipeline.py', 'w') as f:
    f.write(code)
print("done")
