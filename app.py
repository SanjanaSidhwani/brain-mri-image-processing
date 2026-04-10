import streamlit as st
import numpy as np
import cv2
import torch
import json
from pathlib import Path
import tempfile
import sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
import io
from datetime import datetime

matplotlib.use('Agg')

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.volume_utils import load_nifti, zscore_normalize, strip_skull
from src.preprocessing.slice_utils import extract_axial_slices, filter_empty_slices
from src.models.model_factory import create_model
from src.evaluation.gradcam import GradCAM
from src.evaluation.threshold_calibration import load_threshold_for_modality
from src.dataset.input_transforms import build_eval_transform
from src.aggregation.topk_aggregation import robust_patient_prediction_from_tumor_probs
from src.preprocessing.modality_detection import detect_modality


def infer_input_channels_from_state_dict(state_dict):
    for key in ("conv1.0.weight", "module.conv1.0.weight"):
        if key in state_dict:
            return int(state_dict[key].shape[1])
    return 3


EVAL_ALIGNED_AGGREGATION_PARAMS = {
    "threshold": 0.70,
    "top_k": 20,
    "method": "median",
    "min_suspicious_slices": 8,
    "suspicious_prob_threshold": 0.90,
    "min_suspicious_fraction": 0.30,
    "healthy_override_topk_max": 0.20,
    "healthy_override_max_suspicious_slices": 3,
    "healthy_override_max_suspicious_fraction": 0.05,
    "hard_tumor_topk_min": 0.95,
    "hard_tumor_min_suspicious_slices": 3,
}


def get_latest_checkpoint_path(checkpoint_dir="outputs/checkpoints"):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    pth_files = list(checkpoint_dir.glob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    return max(pth_files, key=lambda path: path.stat().st_mtime)


@st.cache_data
def load_aggregation_params(
    config_path="outputs/calibration/aggregation_calibration.json",
    loaded_checkpoint_path: str | None = None,
):
    defaults = EVAL_ALIGNED_AGGREGATION_PARAMS.copy()

    path = Path(config_path)
    if not path.exists():
        return defaults

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        artifact_checkpoint = str(payload.get("checkpoint", "")).replace("\\", "/").lower()
        loaded_checkpoint = str(loaded_checkpoint_path or "").replace("\\", "/").lower()

        # Ignore aggregation artifacts calibrated for a different checkpoint.
        if artifact_checkpoint and loaded_checkpoint and artifact_checkpoint != loaded_checkpoint:
            return defaults

        params = payload.get("best", {}).get("params", {})
        merged = defaults.copy()
        merged.update({k: params[k] for k in defaults.keys() if k in params})
        return merged
    except Exception:
        return defaults


@st.cache_data
def load_slice_threshold(config_path="outputs/calibration/slice_threshold_calibration.json"):
    default_threshold = 0.5
    path = Path(config_path)
    if not path.exists():
        return default_threshold

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        threshold = float(payload.get("threshold", default_threshold))
        if 0.0 <= threshold <= 1.0:
            return threshold
    except Exception:
        pass
    return default_threshold


@st.cache_data
def load_modality_threshold(modality: str):
    return load_threshold_for_modality(modality)


@st.cache_resource
def load_model(checkpoint_path=None):
    if checkpoint_path is None:
        preferred_checkpoint = Path("outputs/checkpoints/step2_balanced_2ep")
        if preferred_checkpoint.exists():
            checkpoint_path = preferred_checkpoint
        else:
            preferred_checkpoint_with_ext = Path("outputs/checkpoints/step2_balanced_2ep.pth")
            if preferred_checkpoint_with_ext.exists():
                checkpoint_path = preferred_checkpoint_with_ext
            else:
                checkpoint_path = get_latest_checkpoint_path()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
    input_channels = infer_input_channels_from_state_dict(state_dict)
    model = create_model(architecture='cnn', num_classes=2, input_channels=input_channels)
    
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    return model, device, input_channels, str(checkpoint_path)


def normalize_slice(slice_2d):
    slice_min = np.min(slice_2d)
    slice_max = np.max(slice_2d)
    if slice_max > slice_min:
        return (slice_2d - slice_min) / (slice_max - slice_min)
    else:
        return np.zeros_like(slice_2d, dtype=np.float32)


def strip_skull_volume(volume_3d):
    return np.stack([strip_skull(volume_3d[:, :, i]) for i in range(volume_3d.shape[2])], axis=2)


def preprocess_slice_for_model(
    slice_2d,
    target_size=224,
    center_crop_size=180,
    input_channels=3,
    prev_slice=None,
    next_slice=None,
):
    eval_transform = build_eval_transform(
        target_size=target_size,
        center_crop_size=center_crop_size,
        num_channels=input_channels,
    )

    if not isinstance(slice_2d, np.ndarray):
        slice_2d = np.asarray(slice_2d)

    slice_2d = slice_2d.astype(np.float32, copy=False)
    if input_channels == 1:
        stacked = np.expand_dims(slice_2d, axis=0)
    elif input_channels == 3:
        if prev_slice is None:
            prev_slice = np.zeros_like(slice_2d, dtype=np.float32)
        if next_slice is None:
            next_slice = np.zeros_like(slice_2d, dtype=np.float32)
        stacked = np.stack(
            [
                np.asarray(prev_slice, dtype=np.float32),
                slice_2d,
                np.asarray(next_slice, dtype=np.float32),
            ],
            axis=0,
        )
    else:
        stacked = np.repeat(np.expand_dims(slice_2d, axis=0), input_channels, axis=0)
    slice_tensor = torch.from_numpy(stacked).float()

    slice_tensor = eval_transform(slice_tensor)

    return slice_tensor


def predict_slices_batch(model, device, slices, max_slices=None, input_channels=3):
    predictions = []
    
    limit = len(slices) if max_slices is None else min(max_slices, len(slices))
    
    for i in range(limit):
        try:
            prev_slice = slices[i - 1] if i > 0 else np.zeros_like(slices[i], dtype=np.float32)
            next_slice = slices[i + 1] if i + 1 < limit else np.zeros_like(slices[i], dtype=np.float32)
            slice_tensor = preprocess_slice_for_model(
                slices[i],
                input_channels=input_channels,
                prev_slice=prev_slice,
                next_slice=next_slice,
            )
            slice_tensor = slice_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(slice_tensor)
                probs = torch.softmax(output, dim=1)
                tumor_prob = probs[0, 1].item()
            
            predictions.append(tumor_prob)
        except Exception as e:
            predictions.append(0.5)
    
    return np.array(predictions)


def predict_slice(model, device, slice_tensor, slice_threshold=0.5):
    slice_tensor = slice_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(slice_tensor)
        probs = torch.softmax(output, dim=1)
        tumor_prob = probs[0, 1].item()
        pred = 1 if tumor_prob >= float(slice_threshold) else 0
        confidence = tumor_prob if pred == 1 else (1.0 - tumor_prob)
    
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


def create_blank_heatmap_and_overlay(normalized_slice):
    base = np.stack([normalized_slice] * 3, axis=-1)
    base = (base * 255).astype(np.uint8) if base.max() <= 1 else base.astype(np.uint8)
    # Keep healthy-case explainability panels visually neutral (no highlighted regions).
    return base.copy(), base.copy()


def create_probability_graph(predictions, theme_mode="dark"):
    is_dark = theme_mode == "dark"
    fig_bg = "#0F172A" if is_dark else "#FFFFFF"
    text_col = "#E2E8F0" if is_dark else "#0F172A"
    grid_col = "#334155" if is_dark else "#CBD5E1"

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(fig_bg)
    ax.set_facecolor(fig_bg)
    
    ax.plot(predictions, linewidth=2.5, color='#3B82F6', label='Tumor Probability')
    ax.fill_between(range(len(predictions)), predictions, alpha=0.22, color='#3B82F6')
    ax.axhline(y=0.5, color='#64748B', linestyle='--', linewidth=1.2, alpha=0.9, label='Decision Threshold')
    
    ax.set_xlabel('Slice Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tumor Probability', fontsize=12, fontweight='bold')
    ax.set_title('Tumor Probability Across Slices', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.35, color=grid_col)
    ax.legend(loc='upper right')

    ax.tick_params(colors=text_col)
    ax.xaxis.label.set_color(text_col)
    ax.yaxis.label.set_color(text_col)
    ax.title.set_color(text_col)
    for spine in ax.spines.values():
        spine.set_color(grid_col)
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return Image.open(buf)


def create_gradcam_panel(normalized_slice, heatmap_rgb, overlay):
    img1 = Image.fromarray((normalized_slice * 255).astype(np.uint8), mode='L')
    img2 = Image.fromarray((heatmap_rgb * 255).astype(np.uint8) if heatmap_rgb.max() <= 1 else heatmap_rgb.astype(np.uint8))
    img3 = Image.fromarray((overlay * 255).astype(np.uint8) if overlay.max() <= 1 else overlay.astype(np.uint8))
    
    size = (350, 350)
    img1 = img1.resize(size)
    img2 = img2.resize(size)
    img3 = img3.resize(size)
    
    panel = Image.new('RGB', (1100, 400), color='white')
    
    img1_rgb = Image.new('RGB', img1.size)
    img1_rgb.paste(img1)
    
    panel.paste(img1_rgb, (10, 25))
    panel.paste(img2, (360, 25))
    panel.paste(img3, (710, 25))
    
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    draw.text((60, 5), "Original MRI", fill='black', font=font)
    draw.text((385, 5), "Grad-CAM Heatmap", fill='black', font=font)
    draw.text((710, 5), "MRI + Grad-CAM Overlay", fill='black', font=font)
    
    return panel


def apply_custom_css():
    css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Source+Sans+3:wght@400;500;600;700&display=swap');

        html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
            font-family: 'Source Sans 3', sans-serif;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Playfair Display', serif !important;
            font-weight: 700;
        }

        .title-main {
            font-family: 'Playfair Display', serif;
            font-size: 48px;
            font-weight: 800;
            color: var(--primary-color, #3B82F6);
            text-align: center;
            margin-bottom: 10px;
        }

        .subtitle {
            font-family: 'Source Sans 3', sans-serif;
            font-size: 18px;
            text-align: center;
            margin-bottom: 20px;
            opacity: 0.9;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="SYNAPSE X",
        page_icon="",
        layout="wide"
    )

    with st.sidebar:
        st.header("Upload & Settings")
        
        uploaded_file = st.file_uploader(
            "Upload MRI file (.nii or .nii.gz)",
            type=["nii", "nii.gz"]
        )
        
        show_gradcam = st.checkbox("Show Grad-CAM Analysis", value=True)
        show_probability_graph = st.checkbox("Show Probability Graph", value=True)
        
        st.markdown("---")
        st.markdown("**Information**")
        st.info(
            "Upload a NIfTI format MRI file for analysis. "
            "SYNAPSE X will predict tumor presence with explainability via Grad-CAM."
        )

    apply_custom_css()

    st.markdown('<div class="title-main">SYNAPSE X</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Brain MRI Analysis with Explainable AI</div>', unsafe_allow_html=True)
    st.markdown("---")
    
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

    gradcam_panel_bytes = None
    prob_graph_bytes = None
    report_pdf_bytes = None
    
    try:
        volume = load_nifti(tmp_path)
        normalized_volume = zscore_normalize(volume)
        normalized_volume = strip_skull_volume(normalized_volume)
        
        slices = extract_axial_slices(normalized_volume)
        slices = filter_empty_slices(slices, threshold=0.05)
        total_slices = len(slices)

        if total_slices == 0:
            st.error("No slices were extracted from the uploaded MRI volume.")
            return
        
        st.subheader("Patient Information")
        st.markdown(f"**Patient ID:** {patient_id}")
        info_col1, info_col2 = st.columns(2)
        with info_col1:
            st.metric("Volume Shape", str(volume.shape))
        with info_col2:
            st.metric("Total Slices", total_slices)
        
        st.markdown("---")

        model, device, input_channels, checkpoint_used = load_model()
        modality = detect_modality(uploaded_file.name)
        slice_threshold = load_modality_threshold(modality)

        all_predictions = predict_slices_batch(model, device, slices, input_channels=input_channels)
        highest_tumor_idx = np.argmax(all_predictions)
        highest_tumor_prob = all_predictions[highest_tumor_idx]

        st.subheader("Patient Diagnosis")
        agg_params = load_aggregation_params(loaded_checkpoint_path=checkpoint_used)
        decision = robust_patient_prediction_from_tumor_probs(
            tumor_probs=all_predictions,
            **agg_params,
        )
        patient_tumor_prob = decision["score"]
        patient_pred = "Tumor Detected" if decision["prediction"] == 1 else "Normal"

        diagnosis_confidence = decision["confidence"]
        diag_col1, diag_col2 = st.columns(2)
        with diag_col1:
            st.metric("Patient-Level Diagnosis", patient_pred)
        with diag_col2:
            st.metric("Confidence", f"{diagnosis_confidence*100:.2f}%")

        st.caption(f"Checkpoint in use: {Path(checkpoint_used).name}")

        st.markdown("---")

        st.subheader("Slice Selection")
        if decision["prediction"] == 1:
            default_slice_index = int(np.argmax(all_predictions))
        else:
            # For healthy predictions, open at the slice with the strongest non-tumor confidence.
            default_slice_index = int(np.argmin(all_predictions))

        slice_index = st.slider(
            "Select Slice",
            min_value=0,
            max_value=total_slices - 1,
            value=default_slice_index
        )

        selected_slice = slices[slice_index]
        normalized_slice = normalize_slice(selected_slice)

        selected_prev = slices[slice_index - 1] if slice_index > 0 else np.zeros_like(selected_slice, dtype=np.float32)
        selected_next = slices[slice_index + 1] if slice_index + 1 < total_slices else np.zeros_like(selected_slice, dtype=np.float32)

        slice_tensor = preprocess_slice_for_model(
            selected_slice,
            input_channels=input_channels,
            prev_slice=selected_prev,
            next_slice=selected_next,
        )
        pred, confidence, tumor_prob = predict_slice(
            model,
            device,
            slice_tensor,
            slice_threshold=slice_threshold,
        )

        st.markdown("---")

        st.subheader("Slice-Level Analysis")
        pred_class = "Tumor" if pred == 1 else "Normal"
        slice_col1, slice_col2, slice_col3, slice_col4, slice_col5 = st.columns(5)

        with slice_col1:
            st.metric("Slice Index", f"{slice_index} / {total_slices - 1}")
        with slice_col2:
            st.metric("Prediction", pred_class)
        with slice_col3:
            st.metric("Confidence", f"{confidence*100:.2f}%")
        with slice_col4:
            st.metric("Tumor Probability", f"{tumor_prob*100:.2f}%")
        with slice_col5:
            st.metric("Highest Tumor Slice", f"#{highest_tumor_idx}")

        st.caption(f"Slice threshold in use ({modality}): {slice_threshold:.3f}")
        
        st.markdown("---")
        
        if show_gradcam:
            st.subheader("Explainability Analysis")

            suppress_gradcam = (decision["prediction"] == 0 and pred == 0)
            if suppress_gradcam:
                heatmap_rgb, overlay = create_blank_heatmap_and_overlay(normalized_slice)
                cam = np.zeros_like(normalized_slice, dtype=np.float32)
            else:
                cam = generate_gradcam(model, device, slice_tensor)
                heatmap_rgb = create_heatmap_rgb(cam)
                overlay = create_overlay(normalized_slice, cam, alpha=0.4)

            display_size = (360, 360)
            original_display = cv2.resize(normalized_slice, display_size, interpolation=cv2.INTER_LINEAR)
            heatmap_display = cv2.resize(heatmap_rgb, display_size, interpolation=cv2.INTER_LINEAR)
            overlay_display = cv2.resize(overlay, display_size, interpolation=cv2.INTER_LINEAR)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Original MRI Slice**")
                st.image(original_display, width="stretch", clamp=True)
            
            with col2:
                st.markdown("**Grad-CAM Heatmap**")
                st.image(heatmap_display, width="stretch")
            
            with col3:
                st.markdown("**MRI + Grad-CAM Overlay**")
                st.image(overlay_display, width="stretch", clamp=True)
            
            if suppress_gradcam:
                st.info("Grad-CAM is suppressed for healthy predictions to avoid false visual highlights.")
            else:
                st.info(
                    "The Grad-CAM heatmap highlights regions the AI focuses on when predicting. "
                    "Brighter colors indicate higher importance for the prediction."
                )
            
            try:
                gradcam_panel = create_gradcam_panel(normalized_slice, heatmap_rgb, overlay)
                
                buf = io.BytesIO()
                gradcam_panel.save(buf, format='PNG')
                buf.seek(0)
                gradcam_panel_bytes = buf.getvalue()
            except Exception as e:
                st.warning(f"Could not generate Grad-CAM panel: {e}")
        
        if show_probability_graph:
            st.markdown("---")
            st.subheader("Tumor Probability Distribution")
            
            try:
                is_dark = "dark" in str(st.get_option("theme.base")).lower()
                prob_graph = create_probability_graph(all_predictions, "dark" if is_dark else "light")
                st.image(prob_graph, width="stretch")
                
                buf_graph = io.BytesIO()
                prob_graph.save(buf_graph, format='PNG')
                buf_graph.seek(0)
                prob_graph_bytes = buf_graph.getvalue()
            except Exception as e:
                st.warning(f"Could not generate probability graph: {e}")
        
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            
            report_buffer = io.BytesIO()
            doc = SimpleDocTemplate(report_buffer, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1f77b4'),
                spaceAfter=12,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#1f77b4'),
                spaceAfter=12,
                spaceBefore=12,
                fontName='Helvetica-Bold'
            )
            
            story.append(Paragraph("SYNAPSE X - Brain MRI Analysis Report", title_style))
            story.append(Spacer(1, 0.2*inch))
            
            story.append(Paragraph("Patient Information", heading_style))
            patient_data = [
                ['Patient ID', patient_id],
                ['Volume Shape', str(volume.shape)],
                ['Total Slices', str(total_slices)],
                ['Analysis Date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            ]
            patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
            patient_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f0f7')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            story.append(patient_table)
            story.append(Spacer(1, 0.2*inch))
            
            story.append(Paragraph("Diagnosis Summary", heading_style))
            diagnosis_data = [
                ['Patient-Level Diagnosis', patient_pred],
                ['Confidence', f'{patient_tumor_prob*100:.2f}%'],
                ['Highest Tumor Slice', f'#{highest_tumor_idx}'],
                ['Highest Tumor Probability', f'{highest_tumor_prob*100:.2f}%']
            ]
            diagnosis_table = Table(diagnosis_data, colWidths=[2*inch, 4*inch])
            diagnosis_color = colors.HexColor('#f8d7da') if patient_pred == "Tumor Detected" else colors.HexColor('#d4edda')
            diagnosis_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f0f7')),
                ('BACKGROUND', (1, 0), (1, -1), diagnosis_color),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            story.append(diagnosis_table)
            story.append(Spacer(1, 0.2*inch))
            
            story.append(Paragraph("Current Slice Analysis", heading_style))
            slice_data = [
                ['Slice Index', f'{slice_index} / {total_slices - 1}'],
                ['Slice Prediction', "Tumor" if pred == 1 else "Normal"],
                ['Slice Confidence', f'{confidence*100:.2f}%'],
                ['Slice Tumor Probability', f'{tumor_prob*100:.2f}%']
            ]
            slice_table = Table(slice_data, colWidths=[2*inch, 4*inch])
            slice_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#e8f0f7')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            story.append(slice_table)
            story.append(Spacer(1, 0.2*inch))

            story.append(Paragraph("Visual Evidence", heading_style))
            if gradcam_panel_bytes is not None:
                gradcam_img = RLImage(io.BytesIO(gradcam_panel_bytes), width=6.5 * inch, height=2.35 * inch)
                story.append(Paragraph("Grad-CAM Panel", styles['Normal']))
                story.append(Spacer(1, 0.08 * inch))
                story.append(gradcam_img)
                story.append(Spacer(1, 0.2 * inch))
            else:
                story.append(Paragraph("Grad-CAM Panel: Not available for this export.", styles['Normal']))
                story.append(Spacer(1, 0.12 * inch))

            if prob_graph_bytes is not None:
                prob_img = RLImage(io.BytesIO(prob_graph_bytes), width=6.5 * inch, height=2.7 * inch)
                story.append(Paragraph("Tumor Probability Distribution", styles['Normal']))
                story.append(Spacer(1, 0.08 * inch))
                story.append(prob_img)
                story.append(Spacer(1, 0.2 * inch))
            else:
                story.append(Paragraph("Probability Graph: Not available for this export.", styles['Normal']))
                story.append(Spacer(1, 0.12 * inch))
            
            story.append(Paragraph("Notes", heading_style))
            story.append(Paragraph(
                "This report was generated by SYNAPSE X, an AI-assisted diagnostic tool. "
                "All results should be reviewed by qualified medical professionals. "
                "The model provides probabilistic predictions based on learned patterns in training data.",
                styles['Normal']
            ))
            
            doc.build(story)
            report_buffer.seek(0)
            report_pdf_bytes = report_buffer.getvalue()
        except ImportError:
            st.warning("ReportLab not installed. PDF download unavailable. Install with: pip install reportlab")
        except Exception as e:
            st.warning(f"Could not generate PDF report: {e}")

        with st.sidebar:
            st.markdown("---")
            st.markdown("**Exports**")
            if report_pdf_bytes is not None:
                st.download_button(
                    label="Download Full Report (PDF)",
                    data=report_pdf_bytes,
                    file_name=f"{patient_id}_report.pdf",
                    mime="application/pdf"
                )
            if gradcam_panel_bytes is not None:
                st.download_button(
                    label="Download Grad-CAM Panel (PNG)",
                    data=gradcam_panel_bytes,
                    file_name=f"{patient_id}_gradcam_{slice_index}.png",
                    mime="image/png"
                )
            if prob_graph_bytes is not None:
                st.download_button(
                    label="Download Probability Graph (PNG)",
                    data=prob_graph_bytes,
                    file_name=f"{patient_id}_probability_graph.png",
                    mime="image/png"
                )
    
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
