import cv2
import numpy as np
import pydicom
from PIL import Image
import os
import datetime
import config

def apply_medical_enhancement(image_np):
    """Standardized Clinical CLAHE for mucosal pattern sharpening."""
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)

def is_diagnostic_quality(image_np):
    """Quality Audit: Rejects images that are clinically non-viable."""
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    if brightness < config.QUALITY_BRIGHTNESS or sharpness < config.QUALITY_SHARPNESS:
        return False, f"Non-Diagnostic: B={brightness:.1f}, S={sharpness:.1f}"
    return True, "Passed"

def load_medical_image(uploaded_file):
    """DICOM & Image Support with Clinical Metadata extraction."""
    if uploaded_file.name.lower().endswith('.dcm'):
        ds = pydicom.dcmread(uploaded_file)
        pixel_array = ds.pixel_array
        if pixel_array.dtype != np.uint8:
            pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        if len(pixel_array.shape) == 2:
            pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(pixel_array).convert('RGB')
        meta = {"PatientID": getattr(ds, "PatientID", "DICOM_PATIENT"), "Date": getattr(ds, "ContentDate", "2026-04-24")}
        return img, meta
    else:
        img = Image.open(uploaded_file).convert('RGB')
        return img, {"PatientID": "External_Image", "Date": "N/A"}

def archive_result(visualization_np, label):
    """Elite Archiving: Ensures BGR/RGB parity and prevents scrambled colors."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"CDSS_{label}_{timestamp}.png"
    filepath = os.path.join(config.RESULTS_DIR, filename)
    
    # Scale float to uint8 correctly
    if visualization_np.dtype != np.uint8:
        save_ready = (visualization_np * 255).astype(np.uint8)
    else:
        save_ready = visualization_np

    # Use PIL to save for absolute color consistency
    final_img = Image.fromarray(save_ready)
    final_img.save(filepath)
    return filename