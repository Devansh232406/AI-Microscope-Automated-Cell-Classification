import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="AI Microscope", page_icon="🔬")

st.title("🔬 AI Microscope: Malaria Cell Classification")
st.write("Upload a microscope cell image to detect malaria infection.")

# ---------------------------
# Device
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load model
# ---------------------------
@st.cache_resource
def load_model():
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
    model.load_state_dict(torch.load("efficientnet_malaria.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# ---------------------------
# Image transform
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------------------------
# Upload image
# ---------------------------
uploaded_file = st.file_uploader("Upload a cell image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()

    if prob > 0.5:
        st.error(f"⚠️ Parasitized Cell Detected (Confidence: {prob:.2f})")
    else:
        st.success(f"✅ Uninfected Cell (Confidence: {1 - prob:.2f})")
