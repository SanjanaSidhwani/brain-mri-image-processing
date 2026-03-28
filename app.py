import streamlit as st
import numpy as np
import cv2
import torch
from pathlib import Path
import tempfile
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.volume_utils import load_nifti, zscore_normalize, strip_skull
from src.preprocessing.slice_utils import extract_axial_slices
from src.models.model_factory import create_model
from src.evaluation.gradcam import GradCAM
from src.dataset.input_transforms import build_eval_transform


@st.cache_resource
def load_model(checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_dir = Path("outputs/checkpoints")
        pth_files = sorted(checkpoint_dir.glob("*.pth"))
        if not pth_files:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        checkpoint_path = pth_files[-1]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(architecture='cnn', num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model, device


def normalize_slice(slice_2d):
    slice_min = np.min(slice_2d)
    slice_max = np.max(slice_2d)
    if slice_max > slice_min:
        return (slice_2d - slice_min) / (slice_max - slice_min)
    else:
        return np.zeros_like(slice_2d, dtype=np.float32)


def strip_skull_volume(volume_3d):
    return np.stack([strip_skull(volume_3d[:, :, i]) for i in range(volume_3d.shape[2])], axis=2)


def preprocess_slice_for_model(slice_2d, target_size=224, center_crop_size=180):
    eval_transform = build_eval_transform(
        target_size=target_size,
        center_crop_size=center_crop_size
    )

    if not isinstance(slice_2d, np.ndarray):
        slice_2d = np.asarray(slice_2d)

    slice_2d = slice_2d.astype(np.float32, copy=False)
    stacked = np.stack([slice_2d, slice_2d, slice_2d], axis=0)
    slice_tensor = torch.from_numpy(stacked).float()

    slice_tensor = eval_transform(slice_tensor)

    return slice_tensor


def predict_slice(model, device, slice_tensor):
    slice_tensor = slice_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(slice_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()
        tumor_prob = probs[0, 1].item()
    
    return pred, confidence, tumor_prob


def generate_gradcam(model, device, slice_tensor):
    slice_tensor = slice_tensor.unsqueeze(0).to(device)
    
    gradcam = GradCAM(model)
    cam = gradcam.generate(slice_tensor, class_idx=1)
    
    return cam


def create_overlay(image_normalized, cam, alpha=0.5):
    if image_normalized.shape != cam.shape:
        image_normalized = cv2.resize(
            image_normalized,
            (cam.shape[1], cam.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    cam_colored = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
    cam_colored = cam_colored / 255.0
    
    image_3ch = np.stack([image_normalized] * 3, axis=-1)
    
    overlay = (1 - alpha) * image_3ch + alpha * cam_colored
    overlay = np.clip(overlay, 0, 1)
    
    return overlay


def create_heatmap_rgb(cam):
    cam_uint8 = (cam * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return heatmap_rgb


def main():
    st.set_page_config(
        page_title="Brain MRI AI Decision Support",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Brain MRI AI Decision Support System")
    st.markdown("### 3D MRI Analysis with Explainable AI (Grad-CAM)")
    st.markdown("---")
    
    with st.sidebar:
        st.header("📁 Upload & Settings")
        
        uploaded_file = st.file_uploader(
            "Upload MRI file (.nii or .nii.gz)",
            type=["nii", "nii.gz"]
        )
        
        show_gradcam = st.toggle("Show Grad-CAM", value=True)
        
        st.markdown("---")
        st.markdown("**Info**")
        st.info(
            "Upload a NIfTI format MRI file to analyze. "
            "The AI will predict tumor presence on selected slices "
            "with explainability via Grad-CAM visualization."
        )
    
    if uploaded_file is None:
        st.warning("Please upload an MRI file to begin analysis.")
        return
    
    file_name_lower = uploaded_file.name.lower()
    if not (file_name_lower.endswith(".nii") or file_name_lower.endswith(".nii.gz")):
        st.error("Invalid file type. Please upload only .nii or .nii.gz files.")
        return

    patient_id = uploaded_file.name.replace(".nii.gz", "").replace(".nii", "")
    temp_suffix = ".nii.gz" if file_name_lower.endswith(".nii.gz") else ".nii"

    with tempfile.NamedTemporaryFile(delete=False, suffix=temp_suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name
    
    try:
        volume = load_nifti(tmp_path)
        normalized_volume = zscore_normalize(volume)
        normalized_volume = strip_skull_volume(normalized_volume)
        
        slices = extract_axial_slices(normalized_volume)
        total_slices = len(slices)

        if total_slices == 0:
            st.error("No slices were extracted from the uploaded MRI volume.")
            return
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Patient ID", patient_id)
        with col2:
            st.metric("Volume Shape", f"{volume.shape}")
        with col3:
            st.metric("Total Slices", total_slices)
        
        st.markdown("---")
        
        slice_index = st.slider(
            "Select Slice",
            min_value=0,
            max_value=total_slices - 1,
            value=total_slices // 2
        )
        
        selected_slice = slices[slice_index]
        normalized_slice = normalize_slice(selected_slice)
        
        model, device = load_model()
        
        slice_tensor = preprocess_slice_for_model(selected_slice)
        pred, confidence, tumor_prob = predict_slice(model, device, slice_tensor)
        
        pred_class = "🔴 Tumor" if pred == 1 else "🟢 Normal"
        pred_color = "red" if pred == 1 else "green"
        
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        with info_col1:
            st.metric("Slice Index", f"{slice_index} / {total_slices - 1}")
        with info_col2:
            st.metric("Prediction", pred_class)
        with info_col3:
            st.metric("Confidence", f"{confidence*100:.2f}%")
        with info_col4:
            st.metric("Tumor Probability", f"{tumor_prob*100:.2f}%")
        
        st.markdown("---")
        
        if show_gradcam:
            st.subheader("📊 Explainability Analysis")
            
            cam = generate_gradcam(model, device, slice_tensor)
            heatmap_rgb = create_heatmap_rgb(cam)
            overlay = create_overlay(normalized_slice, cam, alpha=0.4)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Original MRI Slice**")
                st.image(normalized_slice, width="stretch", clamp=True)
            
            with col2:
                st.markdown("**Grad-CAM Heatmap**")
                st.image(heatmap_rgb, width="stretch")
            
            with col3:
                st.markdown("**Overlay (MRI + Grad-CAM)**")
                st.image(overlay, width="stretch", clamp=True)
            
            st.info(
                "The Grad-CAM heatmap highlights the regions the AI model focuses on "
                "when making predictions. Brighter regions indicate higher importance."
            )
        else:
            st.subheader("📊 MRI Visualization")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original MRI Slice**")
                st.image(normalized_slice, width="stretch", clamp=True)
            
            with col2:
                st.markdown("**Prediction Details**")
                st.write(f"**Predicted Class:** {pred_class}")
                st.write(f"**Confidence Score:** {confidence:.4f}")
                st.write(f"**Tumor Probability:** {tumor_prob:.4f}")
                st.write(f"**Normal Probability:** {1-tumor_prob:.4f}")
    
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
