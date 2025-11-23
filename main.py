# streamlit_xray_app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import io
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "resnet50_xray_finetuned.pth"
target_layer_name = "layer4"

resnet = models.resnet50(pretrained=False)
resnet.fc = nn.Linear(resnet.fc.in_features, 2)
resnet.load_state_dict(torch.load(model_path, map_location=device))
resnet = resnet.to(device)
resnet.eval()

# Grad-CAM setup
target_layer_module = dict([*resnet.named_modules()])[target_layer_name]
cam = GradCAM(model=resnet, target_layers=[target_layer_module])

# ---------------- PREPROCESS ----------------
def apply_mask_to_image(img):
    if img.ndim==3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _,thresh = cv2.threshold(blur,15,255,cv2.THRESH_BINARY)
    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    largest = max(contours,key=cv2.contourArea)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask,[largest],-1,1,cv2.FILLED)
    if img.ndim==3:
        masked = img.copy()
        for c in range(3):
            masked[:,:,c] *= mask
    else:
        masked = img * mask
    return masked

def preprocess_image(img):
    img_masked = apply_mask_to_image(img)
    img_bw = cv2.cvtColor(img_masked, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4)
    clahe_img = np.clip(clahe.apply(img_bw)+30,0,255).astype(np.uint8)
    img_rgb = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
    img_resized = cv2.resize(img_rgb,(224,224), interpolation=cv2.INTER_CUBIC)
    img_normalized = img_resized/255.0
    tensor = torch.from_numpy(img_normalized.transpose(2,0,1)).unsqueeze(0).float()
    return tensor, img_rgb

def generate_overlay(tensor,img_rgb):
    grayscale_cam = cam(input_tensor=tensor)[0]
    img_resized = cv2.resize(img_rgb,(224,224))/255.0
    overlay = show_cam_on_image(np.float32(img_resized),grayscale_cam,use_rgb=True)
    return overlay


# Streamlit
st.set_page_config(page_title="X-ray Pneumonia Classifier", layout="wide")
st.markdown("<h2 style='text-align: center;'>X-ray Pneumonia Classifier with Grad-CAM</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = np.array(Image.open(io.BytesIO(uploaded_file.read())).convert("RGB"))
    
    tensor, img_rgb = preprocess_image(img)
    
    with torch.no_grad():
        output = resnet(tensor.to(device))
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        threshold = 0.6
        pred_label = "PNEUMONIA" if probs[1] > threshold else "NORMAL"
        confidence = probs[1] if pred_label == "PNEUMONIA" else probs[0]
    
    # Grad-CAM
    grayscale_cam = cam(input_tensor=tensor)[0]  # 224x224
    overlay = generate_overlay(tensor, img_rgb)
    
    color = "red" if pred_label == "PNEUMONIA" else "green"

    # Display side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original X-ray")
        st.image(img, use_container_width=True)
    
    with col2:
        st.subheader("Grad-CAM Overlay")
        st.markdown(
            f"<p style='text-align: center; color:{color}; font-size:20px; font-weight:bold;'>Prediction: {pred_label} ({confidence:.2f})</p>",
            unsafe_allow_html=True
        )

        # Create figure with colorbar
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(overlay)
        ax.axis('off')
        
        # Create ScalarMappable for colorbar
        sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        
        # Pass ax explicitly to colorbar
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Intensity", rotation=270, labelpad=15, fontsize=8)
        
        st.pyplot(fig)