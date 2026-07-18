import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import get_model
import pandas as pd

def evaluate_elite_model():
    # 1. Setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    CLASS_NAMES = ['dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 
                   'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis']

    # 2. Test Data Loader
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_set = datasets.ImageFolder('data/processed/test', test_transform)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Load the Best Model
    model = get_model(num_classes=8).to(DEVICE)
    model.load_state_dict(torch.load('models/gastro_sentinel_best.pth'))
    model.eval()

    all_preds = []
    all_labels = []

    print("--- Starting Final Clinical Audit (Test Set) ---")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. CALCULATE PROFESSIONAL METRICS
    cm = confusion_matrix(all_labels, all_preds)
    
    medical_stats = []
    for i in range(len(CLASS_NAMES)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0
        
        medical_stats.append({
            "Class": CLASS_NAMES[i],
            "Sensitivity (Recall)": f"{(sensitivity*100):.2f}%",
            "Specificity": f"{(specificity*100):.2f}%",
            "F1-Score": f"{(f1*100):.2f}%"
        })

    df = pd.DataFrame(medical_stats)
    print("\n--- FINAL CLINICAL METRICS ---")
    print(df.to_string(index=False))

    # --- ELITE MODEL SUMMARY (Calculated inside the function) ---
    print("\n" + "="*35)
    print("      ELITE MODEL GLOBAL SUMMARY")
    print("="*35)
    
    avg_sensitivity = df['Sensitivity (Recall)'].str.rstrip('%').astype(float).mean()
    avg_specificity = df['Specificity'].str.rstrip('%').astype(float).mean()
    avg_f1 = df['F1-Score'].str.rstrip('%').astype(float).mean()
    
    print(f"Total Model Accuracy:    {avg_sensitivity:.2f}%")
    print(f"Total Model Sensitivity: {avg_sensitivity:.2f}%")
    print(f"Total Model Specificity: {avg_specificity:.2f}%")
    print(f"Total Model F1-Score:    {avg_f1:.2f}%")
    print("="*35)

    # 5. GENERATE PROFESSIONAL CONFUSION MATRIX
    plt.figure(figsize=(15, 12)) 
    sns.set_context("paper", font_scale=1.4)
    
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                annot_kws={"size": 13, "weight": "bold"},
                linewidths=0.5, linecolor='gray')
    
    plt.title('GastroSentinel: Clinical Performance Audit', fontsize=22, pad=30, fontweight='bold')
    plt.ylabel('Actual Pathological Diagnosis', fontsize=16, fontweight='bold')
    plt.xlabel('AI Predicted Diagnosis', fontsize=16, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right', fontsize=12) 
    plt.yticks(rotation=0, fontsize=12)
    plt.subplots_adjust(bottom=0.25, left=0.20) 

    plt.savefig('results/metrics/confusion_matrix_final.png', dpi=300, bbox_inches='tight')
    print("\n[SUCCESS] Professional Confusion Matrix saved to: results/metrics/confusion_matrix_final.png")
    plt.show()

if __name__ == "__main__":
    evaluate_elite_model()