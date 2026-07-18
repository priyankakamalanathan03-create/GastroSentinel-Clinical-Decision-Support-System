import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import time
import os
import matplotlib.pyplot as plt

# 1. PAGE CONFIG 
st.set_page_config(page_title="GastroSentinel CDSS", layout="wide")

import config   
import utils    
from model import get_model


from pytorch_grad_cam import GradCAMPlusPlus as GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

st.title("🏥 GastroSentinel: Clinical Decision Support System")

# --- ELITE SIDEBAR: CLINICAL ARCHITECTURE AUDIT ---
st.sidebar.header("🔬 Model Benchmark Audit")

# These numbers match your 'efficientnet-b0_audit.csv', 'resnet-18_audit.csv' 
# and DenseNet clinical summary output EXACTLY.
st.sidebar.table({
    "Architecture": ["DenseNet-121 (Active)", "ResNet-18", "EfficientNet-B0"],
    "Global Sensitivity": ["92.83%", "93.50%", "93.00%"],
    "Polyp Recall": ["98.67%", "97.33%", "97.33%"]
})

st.sidebar.info("""
**Clinical Selection Rationale:**
While ResNet-18 showed a high global average, **DenseNet-121** was selected for production because it achieved the **highest Sensitivity for Malignant Polyps (98.67%)**. 

In Colorectal Cancer screening, minimizing False Negatives for polyps is the primary safety priority. DenseNet's feature concatenation architecture is superior at capturing these subtle mucosal textures.
""")

@st.cache_resource
def load_clinical_engine():
    model = get_model(num_classes=len(config.CLASS_NAMES)).to(config.DEVICE)
    if not os.path.exists(config.MODEL_PATH):
        st.error(f"Model file not found at {config.MODEL_PATH}.")
        return None
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.eval()
    return model

model = load_clinical_engine()

uploaded_file = st.file_uploader("Upload Endoscopy Feed...", type=["jpg", "png", "jpeg", "dcm"])

if uploaded_file and model:
    input_img, metadata = utils.load_medical_image(uploaded_file)
    img_np = np.array(input_img)

    # 2. QUALITY GUARDRAIL
    valid, msg = utils.is_diagnostic_quality(img_np)
    if not valid:
        st.error(f"🛑 NON-DIAGNOSTIC: {msg}")
    else:
        # 3. ENHANCEMENT & INFERENCE
        enhanced_np = utils.apply_medical_enhancement(img_np)
        enhanced_img = Image.fromarray(enhanced_np)
        
        transform = transforms.Compose([
            transforms.Resize(config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(enhanced_img).unsqueeze(0).to(config.DEVICE)

        start = time.time()
        with torch.no_grad():
            output = model(input_tensor)
            probs = F.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
        latency = (time.time() - start) * 1000

        # 4. SAFETY FILTER 
        if conf.item() < config.SAFETY_THRESHOLD:
            st.error(f"⚠️ UNCERTAIN ({conf.item()*100:.1f}%): System cannot verify GI markers.")
        else:
            # 5. SHARP XAI VISUALIZATION
            target_layers = [model.backbone.features[-1]]
            cam = GradCAM(model=model, target_layers=target_layers)
            grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(pred.item())])[0, :]
            
            # Overlay on ENHANCED image for sharpest pathological context
            viz_bg = np.array(enhanced_img.resize(config.IMG_SIZE)) / 255.0
            visualization = show_cam_on_image(viz_bg, grayscale_cam, use_rgb=True, image_weight=0.6)

            # 6. AUTO-ARCHIVE
            saved_id = utils.archive_result(visualization, config.CLASS_NAMES[pred.item()])

            # MAIN VIEW
            col1, col2 = st.columns(2)
            with col1: st.image(enhanced_img, caption="Clinical Feed (Enhanced)", use_column_width=True)
            with col2: st.image(visualization, caption="Pathological ROI (Sharp Grad-CAM++)", use_column_width=True)

            # 7. PROBABILITY GRAPH
            st.markdown("### 📊 Diagnostic Certainty Distribution")
            prob_vals = [probs[0][i].item() for i in range(8)]
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = ['#1f77b4']*8; colors[pred.item()] = '#d62728' 
            ax.bar(config.CLASS_NAMES, prob_vals, color=colors, edgecolor='black')
            ax.set_ylim(0, 1.1)
            plt.xticks(rotation=30, ha='right', fontsize=9)
            st.pyplot(fig)

            # 8. CLINICAL SIDEBAR
            st.sidebar.header("📋 Clinical Report")
            st.sidebar.write(f"Patient ID: **{metadata['PatientID']}**")
            st.sidebar.write(f"Study Date: **{metadata['Date']}**")
            st.sidebar.divider()
            st.sidebar.info(f"Diagnosis: **{config.CLASS_NAMES[pred.item()]}**")
            st.sidebar.metric("Latency", f"{latency:.2f} ms", "Real-time Ready")
            st.sidebar.write(f"Certainty: **{conf.item()*100:.1f}%**")
            
            if conf.item() > config.HIGH_CONFIDENCE:
                st.sidebar.success("✅ Verified Findings")
            else:
                st.sidebar.warning("⚠️ Review Needed")
            
            st.sidebar.caption(f"Archived ID: {saved_id}")