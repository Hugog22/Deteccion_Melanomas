import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.model import MelanomaClassifier
from src.data_pipeline import ISICDataset
from run_pipeline import CONFIG

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    model = MelanomaClassifier(freeze_strategy=CONFIG["freeze_strategy"])
    ckpt_path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state_dict"])
    model.to(device)
    model.eval()
    
    df = pd.read_csv(os.path.join(CONFIG["data_dir"], "metadata.csv"))
    test_df = df[df["split"] == "test"].copy()
    
    test_dataset = ISICDataset(test_df, split="test", img_size=CONFIG["img_size"], condition="partial")
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    diffs = []
    flips = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig).squeeze()
            preds_orig = (probs_orig > 0.5).int()
            
            images_occ = images.clone()
            # Occlude the top-right quarter roughly (x: 120-224, y: 0-100)
            images_occ[:, :, :100, 120:] = 0.0
            
            logits_occ = model(images_occ)
            probs_occ = torch.sigmoid(logits_occ).squeeze()
            preds_occ = (probs_occ > 0.5).int()
            
            if probs_orig.dim() == 0:
                probs_orig = probs_orig.unsqueeze(0)
                probs_occ = probs_occ.unsqueeze(0)
                preds_orig = preds_orig.unsqueeze(0)
                preds_occ = preds_occ.unsqueeze(0)
                
            diff = torch.abs(probs_orig - probs_occ).cpu().numpy()
            diffs.extend(diff)
            
            flips += (preds_orig != preds_occ).sum().item()
            total += images.size(0)
            
    mean_diff = np.mean(diffs)
    max_diff = np.max(diffs)
    
    print("=== RESULTADOS DEL TEST DE OCLUSIÓN ===")
    print(f"Total imágenes evaluadas: {total}")
    print(f"Cambio medio en probabilidad: {mean_diff:.4f} ({mean_diff*100:.2f}%)")
    print(f"Cambio máximo en probabilidad: {max_diff:.4f} ({max_diff*100:.2f}%)")
    print(f"Predicciones que cambiaron de clase (flips): {flips} de {total} ({(flips/total)*100:.2f}%)")

if __name__ == "__main__":
    main()
