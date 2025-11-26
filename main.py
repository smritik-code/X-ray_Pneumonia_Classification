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
import segmentation_models_pytorch as smp

device = 'cuda' if torch.cuda.is_available() else 'cpu'

resnet_model_path = "resnet50_xray_finetuned.pth"
resnet = models.resnet50(pretrained=False)
resnet.fc = nn.Linear(resnet.fc.in_features, 2)
resnet.load_state_dict(torch.load(resnet_model_path, map_location=device))
resnet = resnet.to(device)
resnet.eval()

target_layer = dict([*resnet.named_modules()])['layer4']
cam = GradCAM(model=resnet, target_layers=[target_layer])

unet_model_path = "unet_resnet50_xray_seg.pth"
unet = smp.Unet(
    encoder_name="resnet50",
    encoder_weights=None,
    in_channels=3,
    classes=1
)
unet.load_state_dict(torch.load(unet_model_path, map_location=device))
unet = unet.to(device)
unet.eval()

def preprocess_image(img, target_size=(224,224)):
    img_resized = cv2.resize(img, target_size)
    img_normalized = img_resized / 255.0
    tensor = torch.from_numpy(img_normalized.transpose(2,0,1)).unsqueeze(0).float()
    return tensor, img_resized

def generate_mask_overlay(original_img, mask_pred):
    mask_np = mask_pred.squeeze().cpu().numpy()
    mask_np = (mask_np > 0.5).astype(np.uint8)
    mask_resized = cv2.resize(mask_np, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_colored = np.zeros_like(original_img)
    mask_colored[:,:,0] = mask_resized * 255
    overlay = cv2.addWeighted(original_img, 0.7, mask_colored, 0.3, 0)
    return overlay

def generate_gradcam_overlay(original_img, input_tensor):
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    cam_resized = cv2.resize(grayscale_cam, (original_img.shape[1], original_img.shape[0]))
    overlay = show_cam_on_image(np.float32(original_img/255.0), cam_resized, use_rgb=True)
    return overlay, cam_resized

st.set_page_config(page_title="X-ray Pneumonia Detection & Segmentation", layout="wide")
st.markdown("<h2 style='text-align: center;'>X-ray Pneumonia Detection & Segmentation</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = np.array(Image.open(io.BytesIO(uploaded_file.read())).convert("RGB"))
    tensor, _ = preprocess_image(img)

    with torch.no_grad():
        output = resnet(tensor.to(device))
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_label = "PNEUMONIA" if probs[1] > 0.6 else "NORMAL"

    st.markdown(f"<h3 style='text-align:center; color:red;'>Prediction: {pred_label} ({probs[1]:.2f})</h3>", unsafe_allow_html=True)


margin_left, content_col, margin_right = st.columns([0.8,2,0.8])

with content_col:
    left_col, right_col = st.columns([1,1])

    with left_col:
        st.subheader("Original X-ray")
        st.image(img, use_container_width=True)

    with right_col:
        # Grad-CAM
        st.subheader("Grad-CAM")
        gradcam_overlay, cam_resized = generate_gradcam_overlay(img, tensor)
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(gradcam_overlay)
        ax.axis('off')
        sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Intensity", rotation=270, labelpad=15, fontsize=10)
        st.pyplot(fig, use_container_width=True)

        # U-Net 
        if pred_label == "PNEUMONIA":
            with torch.no_grad():
                mask_pred = unet(tensor.to(device))
            overlay_mask = generate_mask_overlay(img, mask_pred)
            st.subheader("U-Net Segmentation Overlay")
            st.image(overlay_mask, use_container_width=True)
