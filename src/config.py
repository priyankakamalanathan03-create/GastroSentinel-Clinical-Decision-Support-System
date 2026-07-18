import torch
import os

# 1. DIRECTORY STRUCTURE
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(BASE_DIR, ".."))

# Main Paths
MODEL_PATH  = os.path.join(ROOT_DIR, "models", "gastro_sentinel_best.pth")
DATA_PROC   = os.path.join(ROOT_DIR, "data", "processed")
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "heatmaps")
METRICS_DIR = os.path.join(ROOT_DIR, "results", "metrics")

# Ensure directories exist
for folder in [os.path.join(ROOT_DIR, "models"), RESULTS_DIR, METRICS_DIR]:
    os.makedirs(folder, exist_ok=True)

# 2. HARDWARE & CLINICAL CLASSES
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = [
    'dyed-lifted-polyps', 'dyed-resection-margins', 'esophagitis', 
    'normal-cecum', 'normal-pylorus', 'normal-z-line', 'polyps', 'ulcerative-colitis'
]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = (224, 224)

# 3. CLINICAL HYPERPARAMETERS (Identical for all 3 models)
EPOCHS = 25
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
LABEL_SMOOTH = 0.1
RANDOM_SEED = 42

# ImageNet Standard Normalization
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]

# 4. SAFETY GUARDRAILS (For Streamlit app)
SAFETY_THRESHOLD = 0.75  
HIGH_CONFIDENCE  = 0.90   
QUALITY_BRIGHTNESS = 25  
QUALITY_SHARPNESS = 100