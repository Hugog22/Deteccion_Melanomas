import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from src.model import MelanomaClassifier
from src.data_pipeline import ISICDataset
from run_pipeline import CONFIG

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = MelanomaClassifier(freeze_strategy=CONFIG["freeze_strategy"])
    ckpt = torch.load(os.path.join(CONFIG["checkpoint_dir"], "best_model.pth"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    
    df = pd.read_csv(os.path.join(CONFIG["data_dir"], "metadata.csv"))
    test_df = df[df["split"] == "test"].copy()
    test_dataset = ISICDataset(test_df, split="test", img_size=224, condition="partial")
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    diffs_tr = []
    diffs_tl = []
    diffs_br = []
    diffs_bl = []
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            logits_orig = model(images)
            probs_orig = torch.sigmoid(logits_orig).squeeze()
            if probs_orig.dim() == 0: probs_orig = probs_orig.unsqueeze(0)
            
            # Top-Right mask
            img_tr = images.clone(); img_tr[:, :, :64, -64:] = 0
            probs_tr = torch.sigmoid(model(img_tr)).squeeze()
            if probs_tr.dim() == 0: probs_tr = probs_tr.unsqueeze(0)
            
            # Top-Left mask
            img_tl = images.clone(); img_tl[:, :, :64, :64] = 0
            probs_tl = torch.sigmoid(model(img_tl)).squeeze()
            if probs_tl.dim() == 0: probs_tl = probs_tl.unsqueeze(0)
            
            # Bottom-Right mask
            img_br = images.clone(); img_br[:, :, -64:, -64:] = 0
            probs_br = torch.sigmoid(model(img_br)).squeeze()
            if probs_br.dim() == 0: probs_br = probs_br.unsqueeze(0)

            # Bottom-Left mask
            img_bl = images.clone(); img_bl[:, :, -64:, :64] = 0
            probs_bl = torch.sigmoid(model(img_bl)).squeeze()
            if probs_bl.dim() == 0: probs_bl = probs_bl.unsqueeze(0)
            
            diffs_tr.extend(torch.abs(probs_orig - probs_tr).cpu().numpy())
            diffs_tl.extend(torch.abs(probs_orig - probs_tl).cpu().numpy())
            diffs_br.extend(torch.abs(probs_orig - probs_br).cpu().numpy())
            diffs_bl.extend(torch.abs(probs_orig - probs_bl).cpu().numpy())
            
    print("=== CONTROLLED OCCLUSION SENSITIVITY (Mean Abs Drop) ===")
    print(f"Top-Left:     {np.mean(diffs_tl):.4f}")
    print(f"Top-Right:    {np.mean(diffs_tr):.4f}  <-- If XAI bias is real model bias, this should be huge")
    print(f"Bottom-Left:  {np.mean(diffs_bl):.4f}")
    print(f"Bottom-Right: {np.mean(diffs_br):.4f}")

if __name__ == "__main__":
    main()
