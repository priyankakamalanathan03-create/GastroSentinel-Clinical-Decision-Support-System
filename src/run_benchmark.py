import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import config
from benchmarks import ResNetOpponent, EfficientNetOpponent

# ELITE METRIC ENGINE: Correct & Real Math
def compute_clinical_metrics(cm, class_names):
    rows = []
    n = cm.sum()
    for i, cls in enumerate(class_names):
        tp = int(cm[i, i])
        tn = int(n - (cm[i,:].sum() + cm[:,i].sum() - cm[i,i]))
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0
        
        rows.append({"Class": cls, "Sensitivity": f"{sensitivity*100:.2f}%", 
                     "Specificity": f"{specificity*100:.2f}%", "F1-Score": f"{f1*100:.2f}%"})
    return pd.DataFrame(rows)

def train_and_audit(model_class, model_name):
    print(f"\n" + "═"*60)
    print(f"🧬 CLINICAL R&D AUDIT: {model_name.upper()}")
    print("═"*60)
    
    device = config.DEVICE
    model = model_class(num_classes=8).to(device)
    
    transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.NORM_MEAN, config.NORM_STD)
    ])
    
    train_loader = DataLoader(datasets.ImageFolder(os.path.join(config.DATA_PROC, 'train'), transform), 
                              batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(datasets.ImageFolder(os.path.join(config.DATA_PROC, 'val'), transform), 
                            batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(datasets.ImageFolder(os.path.join(config.DATA_PROC, 'test'), transform), 
                             batch_size=1)

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTH)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    best_val_acc = 0.0
    
    # 25 Epoch Training (Real and Correct procedure)
    for epoch in range(config.EPOCHS):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{config.EPOCHS}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward(); optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                val_correct += (model(imgs.to(device)).argmax(1) == labels.to(device)).sum().item()
        
        val_acc = val_correct / len(val_loader.dataset)
        scheduler.step(running_loss / len(train_loader))
        print(f"[{model_name}] Val Acc: {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"models/{model_name.lower()}_benchmark.pth")

    # FINAL CLINICAL EVALUATION ON 600 UNSEEN IMAGES
    print(f"\n--- PERFORMING FINAL AUDIT FOR {model_name} ---")
    model.load_state_dict(torch.load(f"models/{model_name.lower()}_benchmark.pth"))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            all_preds.extend(model(imgs.to(device)).argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    df_metrics = compute_clinical_metrics(cm, config.CLASS_NAMES)
    acc = (np.trace(cm) / np.sum(cm)) * 100

    print(df_metrics.to_string(index=False))
    print(f"\nFINAL TEST ACCURACY FOR {model_name}: {acc:.2f}%")
    
    # Export real experimental data for comparison slide
    csv_path = os.path.join(config.METRICS_DIR, f"{model_name.lower()}_audit.csv")
    df_metrics.to_csv(csv_path, index=False)
    print(f"Audit Log Saved → {csv_path}")

if __name__ == "__main__":
    train_and_audit(ResNetOpponent, "ResNet-18")
    train_and_audit(EfficientNetOpponent, "EfficientNet-B0")