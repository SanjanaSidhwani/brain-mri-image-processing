import sys
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_factory import create_model
from src.evaluation.threshold_calibration import load_threshold_for_modality
from src.preprocessing.modality_detection import detect_modality
from src.preprocessing.volume_utils import load_nifti, zscore_normalize, strip_skull
from src.preprocessing.slice_utils import extract_axial_slices
from src.dataset.input_transforms import build_eval_transform
from src.aggregation.topk_aggregation import robust_patient_prediction_from_tumor_probs


def _infer_input_channels_from_state_dict(state_dict) -> int:
    keys = ["conv1.0.weight", "module.conv1.0.weight"]
    for key in keys:
        if key in state_dict:
            return int(state_dict[key].shape[1])
    return 3


def get_latest_checkpoint(checkpoint_dir: str = "outputs/checkpoints") -> str:
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    pth_files = sorted(checkpoint_dir.glob("*.pth"))
    if not pth_files:
        raise FileNotFoundError(f"No .pth files found in {checkpoint_dir}")
    
    return str(pth_files[-1])


def load_aggregation_params(config_path: str = "outputs/calibration/aggregation_calibration.json") -> dict:
    defaults = {
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

    path = Path(config_path)
    if not path.exists():
        return defaults

    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        best = payload.get("best", {})
        params = best.get("params", {})
        merged = defaults.copy()
        merged.update({k: params[k] for k in defaults.keys() if k in params})
        return merged
    except Exception:
        return defaults


def predict_on_mri(
    mri_path: str,
    checkpoint_path: str = None,
    device: str = "auto",
    target_size: int = 224,
    center_crop_size: int = 180,
    apply_skull_strip: bool = True
) -> dict:
    
    if checkpoint_path is None:
        checkpoint_path = get_latest_checkpoint()
    
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    mri_path = Path(mri_path)
    if not mri_path.exists():
        raise FileNotFoundError(f"MRI file not found: {mri_path}")
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    modality = detect_modality(str(mri_path))

    volume = load_nifti(str(mri_path))
    normalized_volume = zscore_normalize(volume)
    
    if apply_skull_strip:
        normalized_volume = np.stack(
            [strip_skull(normalized_volume[:, :, i]) for i in range(normalized_volume.shape[2])],
            axis=2,
        )
    
    slices = extract_axial_slices(normalized_volume)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    input_channels = _infer_input_channels_from_state_dict(state_dict)

    eval_transform = build_eval_transform(
        target_size=target_size,
        center_crop_size=center_crop_size,
        num_channels=input_channels,
    )

    def _preprocess_slice(slice_2d):
        slice_2d = np.asarray(slice_2d, dtype=np.float32)
        if input_channels == 1:
            stacked = np.expand_dims(slice_2d, axis=0)
        else:
            stacked = np.repeat(np.expand_dims(slice_2d, axis=0), input_channels, axis=0)
        tensor = torch.from_numpy(stacked.astype(np.float32)).float()
        return eval_transform(tensor)

    model = create_model(architecture='cnn', num_classes=2, input_channels=input_channels)
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    
    predictions_all = []
    confidences_all = []
    tumor_probs_all = []
    slice_threshold = load_threshold_for_modality(modality)
    
    with torch.no_grad():
        for slice_2d in slices:
            slice_tensor = _preprocess_slice(slice_2d)
            
            if slice_tensor.ndim == 2:
                slice_tensor = slice_tensor.unsqueeze(0)
            
            if slice_tensor.dim() == 3 and slice_tensor.shape[0] != input_channels:
                if slice_tensor.shape[-1] == input_channels:
                    slice_tensor = slice_tensor.permute(2, 0, 1)
                else:
                    slice_tensor = slice_tensor[:1].repeat(input_channels, 1, 1)
            
            slice_tensor = slice_tensor.unsqueeze(0).to(device)
            
            output = model(slice_tensor)
            probs = F.softmax(output, dim=1)
            pred = 1 if probs[0, 1].item() >= float(slice_threshold) else 0
            conf = probs[0, pred].item()
            tumor_prob = probs[0, 1].item()
            
            predictions_all.append(pred)
            confidences_all.append(conf)
            tumor_probs_all.append(tumor_prob)
    
    avg_tumor_prob = np.mean(tumor_probs_all)
    tumor_slices = sum(1 for p in predictions_all if p == 1)
    decision = robust_patient_prediction_from_tumor_probs(
        tumor_probs=tumor_probs_all,
        **load_aggregation_params(),
    )
    final_pred = decision["prediction"]
    final_confidence = decision["confidence"]
    
    return {
        "prediction": final_pred,
        "confidence": final_confidence,
        "tumor_probability": decision["risk_score"],
        "topk_tumor_probability": decision["score"],
        "mean_tumor_probability": float(avg_tumor_prob),
        "tumor_slices": tumor_slices,
        "suspicious_slices": decision["suspicious_slices"],
        "suspicious_fraction": decision["suspicious_fraction"],
        "total_slices": len(predictions_all),
        "checkpoint_used": str(checkpoint_path),
        "modality": modality,
        "slice_threshold_used": float(slice_threshold),
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference on MRI file")
    parser.add_argument("mri_path", type=str, help="Path to MRI file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--crop_size", type=int, default=180, help="Center crop size")
    
    args = parser.parse_args()
    
    result = predict_on_mri(
        args.mri_path,
        checkpoint_path=args.checkpoint,
        device=args.device,
        center_crop_size=args.crop_size
    )
    
    print("\n" + "=" * 60)
    print("Inference Results")
    print("=" * 60)
    print(f"Prediction: {'Tumor (1)' if result['prediction'] == 1 else 'Normal (0)'}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Tumor Probability: {result['tumor_probability']:.4f}")
    print(f"Tumor Slices: {result['tumor_slices']} / {result['total_slices']}")
    print(f"Detected modality: {result['modality']}")
    print(f"Slice threshold used: {result['slice_threshold_used']:.4f}")
    print(f"Checkpoint: {result['checkpoint_used']}")
    print("=" * 60 + "\n")
