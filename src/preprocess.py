import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def apply_medical_enhancement(image):
    #  Medical CLAHE (Contrast Enhancement)
    # Optimized for Gastrointestinal Tissue textures
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

def process_and_split():
    raw_dir = "data/raw"
    base_output = "data/processed"
    classes = os.listdir(raw_dir)
    
    # Ratios for Medical Rigor: 70% Train, 15% Val, 15% Test
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    print(f"Starting Elite Pre-processing [70/15/15 Split] for {len(classes)} classes...")

    for cls in classes:
        cls_path = os.path.join(raw_dir, cls)
        images = os.listdir(cls_path)
        
        # 1. Split off 70% for Training
        train_idx, temp_idx = train_test_split(images, test_size=(VAL_RATIO + TEST_RATIO), random_state=42)
        
        # 2. Split the remaining 30% into 15% Val and 15% Test
        val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)
        
        for subset_idx, subset_name in zip([train_idx, val_idx, test_idx], ['train', 'val', 'test']):
            output_path = os.path.join(base_output, subset_name, cls)
            os.makedirs(output_path, exist_ok=True)
            
            print(f"Enhancing {subset_name} set for {cls}...")
            for img_name in tqdm(subset_idx):
                img_path = os.path.join(cls_path, img_name)
                img = cv2.imread(img_path)
                
                if img is not None:
                    # 1. Standardize Resolution
                    img = cv2.resize(img, (224, 224))
                    # 2. Medical Texture Enhancement
                    img = apply_medical_enhancement(img)
                    # 3. Save to Processed Directory
                    cv2.imwrite(os.path.join(output_path, img_name), img)

if __name__ == "__main__":
    process_and_split()
    print("\n[SUCCESS] Elite Pre-processing Complete.")
    print("Split Ratios: 70% Train | 15% Validation | 15% Test (Hold-out)")