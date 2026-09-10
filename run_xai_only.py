import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.model import MelanomaClassifier
from src.data_pipeline import get_transforms, apply_circular_crop
from src.xai_gradcam import CleverHansAnalyzer, GradCAMPlusPlus
from run_pipeline import CONFIG

def main():
    print("Recalculando mapas Grad-CAM++ sin re-entrenar...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 1. Cargar el modelo ya entrenado
    model = MelanomaClassifier(freeze_strategy=CONFIG["freeze_strategy"])
    ckpt_path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"No se encontró el modelo en {ckpt_path}. ¡Debes entrenar primero!")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 2. Configurar analizador
    target_layer = model.get_features_layer()
    analyzer = CleverHansAnalyzer(
        model=model, target_layer=target_layer,
        device=device, use_gradcam_pp=True
    )
    
    # 3. Cargar datos
    df = pd.read_csv(os.path.join(CONFIG["data_dir"], "metadata.csv"))
    test_df = df[df["split"] == "test"].copy()
    
    transform_test = get_transforms("test", CONFIG["img_size"], condition="partial")
    
    # Selección BALANCEADA
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
    
    xai_samples = []
    for _, row in xai_selection.iterrows():
        try:
            img = Image.open(row["image_path"]).convert("RGB")
            img_np = np.array(img)
            img_np = apply_circular_crop(img_np)
            img_tensor = transform_test(image=img_np)["image"]
            artifact_type = row.get("artifact_type", "none")
            xai_samples.append((img_tensor, row["label"], artifact_type, row["image_path"]))
        except Exception as e:
            print(f"Error cargando imagen: {e}")
            
    if xai_samples:
        os.makedirs(CONFIG["results_dir"], exist_ok=True)
        save_path = os.path.join(CONFIG["results_dir"], "gradcam_analysis_fixed.png")
        fig_xai = analyzer.visualize_clever_hans_effect(
            xai_samples,
            save_path=save_path,
            figsize=(22, 5)
        )
        print(f"✅ ¡Mapas corregidos generados en: {save_path}!")
    else:
        print("⚠️ No hay muestras para Grad-CAM.")

if __name__ == "__main__":
    main()
