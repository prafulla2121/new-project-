# app.py
import streamlit as st
from PIL import Image
import torch
from torchvision import transforms, models
import numpy as np
import io
import os
import pickle
import pyttsx3

st.set_page_config("ISL Phrase Classifier", layout="wide")

# -------------------------
# Helper: load class names
# -------------------------
def load_class_names(label_encoder_path="label_encoder.pkl", dataset_dir="datasets/ISL_Phrases/train"):
    # 1) try to load label encoder
    if os.path.exists(label_encoder_path):
        try:
            with open(label_encoder_path, "rb") as f:
                le = pickle.load(f)
                classes = list(le.classes_)
                return classes
        except Exception as e:
            st.warning(f"Could not load label encoder: {e}")

    # 2) try to read folder names under dataset_dir
    if os.path.exists(dataset_dir):
        try:
            folders = [d for d in sorted(os.listdir(dataset_dir)) if os.path.isdir(os.path.join(dataset_dir, d))]
            if len(folders) > 0:
                return folders
        except Exception as e:
            st.warning(f"Could not read dataset folders: {e}")

    # 3) fallback default (small example) — edit to match your dataset if needed
    return [
        "meet","mistake","open","opinion","pass","please","practice","pressure","problem",
        "questions","remember","seat","shift","sick","stop","sun","team","thirsty","this",
        "together","understand","wait","where","write"
    ]

# -------------------------
# Caching: load model
# -------------------------
@st.cache_resource
def load_model_and_transform(model_path="isl_classifier.pth", num_classes=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # build model architecture (ResNet18)
    model = models.resnet18(pretrained=False)
    if num_classes is None:
        # temporary placeholder -> will be replaced after calling load_class_names
        num_classes = 1000
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    # load weights
    if os.path.exists(model_path):
        try:
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state)
        except Exception as e:
            st.error(f"Failed to load model weights: {e}")
    else:
        st.warning(f"Model not found at {model_path}. Please place isl_classifier.pth in the app folder.")
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return model, transform, device

# -------------------------
# Prediction helper
# -------------------------
def predict_pil_image(pil_img, model, transform, device, class_names):
    img = pil_img.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        probs = torch.nn.functional.softmax(out, dim=1)
        conf, pred = torch.max(probs, 1)
        label = class_names[pred.item()] if pred.item() < len(class_names) else f"class_{pred.item()}"
        return label, float(conf.cpu().item())

# -------------------------
# UI
# -------------------------
st.title("Indian Sign Language — Phrase Classifier")
st.write("Upload an image or use the camera to capture a phrase. The model predicts the phrase and can speak it.")

# load class names first (so we know num_classes)
class_names = load_class_names()
num_classes = len(class_names)

# load model (cached)
model, transform, device = load_model_and_transform(model_path="isl_classifier.pth", num_classes=num_classes)

# If the loaded model architecture had different num_classes, reassign fc if needed:
if model.fc.out_features != num_classes:
    # recreate fc and attempt to load weights again gracefully
    try:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        # try to reload weights (in case saved dict matches)
        if os.path.exists("isl_classifier.pth"):
            state = torch.load("isl_classifier.pth", map_location=device)
            model.load_state_dict(state)
    except Exception:
        # If reload fails, model will still run but prediction indices may not align perfectly
        pass
model.eval()

# TTS engine (init once)
try:
    tts_engine = pyttsx3.init()
except Exception:
    tts_engine = None

# Sidebar options
st.sidebar.header("Settings")
do_speak = st.sidebar.checkbox("Enable text-to-speech (speak prediction)", value=False)
confidence_threshold = st.sidebar.slider("Speak only if confidence ≥", 0.0, 1.0, 0.2)

# Two-column layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Image")
    uploaded = st.file_uploader("Upload JPG / PNG image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        try:
            image = Image.open(uploaded)
            st.image(image, caption="Uploaded image", use_column_width=True)
            if st.button("Classify uploaded image"):
                label, conf = predict_pil_image(image, model, transform, device, class_names)
                st.success(f"Prediction: **{label}**  (confidence: {conf:.2f})")
                if do_speak and tts_engine and conf >= confidence_threshold:
                    try:
                        tts_engine.say(label)
                        tts_engine.runAndWait()
                    except Exception as e:
                        st.warning(f"TTS failed: {e}")
        except Exception as e:
            st.error(f"Could not process uploaded image: {e}")

with col2:
    st.subheader("Camera (snapshot)")
    st.write("Press 'Take photo' then click 'Classify camera image' to predict.")
    cam_image = st.camera_input("Take a photo")
    if cam_image is not None:
        try:
            img_pil = Image.open(io.BytesIO(cam_image.getvalue()))
            st.image(img_pil, caption="Camera capture", use_column_width=True)
            if st.button("Classify camera image"):
                label, conf = predict_pil_image(img_pil, model, transform, device, class_names)
                st.success(f"Prediction: **{label}**  (confidence: {conf:.2f})")
                if do_speak and tts_engine and conf >= confidence_threshold:
                    try:
                        tts_engine.say(label)
                        tts_engine.runAndWait()
                    except Exception as e:
                        st.warning(f"TTS failed: {e}")
        except Exception as e:
            st.error(f"Could not process camera image: {e}")

st.markdown("---")
st.subheader("Model / Classes info")
st.write(f"Number of classes: **{num_classes}**")
# collapse long class lists
with st.expander("Show class names"):
    st.write(class_names)

st.caption("Notes: \n• Ensure `isl_classifier.pth` (PyTorch state_dict) and optionally `label_encoder.pkl` are present in app folder. \n• The camera in Streamlit captures snapshots; for continuous live streaming you can use streamlit-webrtc (more setup).")
