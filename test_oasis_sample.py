import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.volume_utils import load_nifti, zscore_normalize, strip_skull
from src.preprocessing.slice_utils import extract_axial_slices
from src.models.model_factory import create_model
from src.dataset.input_transforms import build_eval_transform
from src.aggregation.topk_aggregation import robust_patient_prediction_from_tumor_probs
from src.inference import load_aggregation_params


def find_oasis_sample() -> str:
    root = PROJECT_ROOT / "data" / "raw" / "oasis"
    if not root.exists():
        raise FileNotFoundError(f"OASIS root not found: {root}")

    for pattern in ("*.nii.gz", "*.nii"):
        matches = sorted(root.rglob(pattern))
        if matches:
            return str(matches[0])

    raise FileNotFoundError(f"No OASIS NIfTI file found under {root}")


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


def predict_slices_batch(model, device, slices, max_slices=None):
    if max_slices is not None:
        slices = slices[:max_slices]
    
    predictions = []
    probabilities = []

    with torch.no_grad():
        for slice_2d in slices:
            slice_tensor = preprocess_slice_for_model(slice_2d)
            slice_tensor = slice_tensor.unsqueeze(0).to(device)
            
            logits = model(slice_tensor)
            probs = torch.softmax(logits, dim=1)
            
            pred_class = torch.argmax(probs, dim=1).item()
            pred_prob = probs[0].cpu().numpy()
            
            predictions.append(pred_class)
            probabilities.append(pred_prob)

    return predictions, probabilities


def main():
    print("=" * 60)
    print("OASIS Sample Evaluation")
    print("=" * 60)
    
    oasis_path = find_oasis_sample()
    print(f"\nLoading OASIS volume: {oasis_path}")
    
    try:
        volume = load_nifti(oasis_path)
        print(f"✓ Loaded OASIS volume | Shape: {volume.shape}")
    except Exception as e:
        print(f"✗ Failed to load volume: {e}")
        return
    
    print("\nPreprocessing volume...")
    try:
        volume = zscore_normalize(volume)
        print(f"✓ Z-score normalized")
        
        volume = np.stack(
            [strip_skull(volume[:, :, i]) for i in range(volume.shape[2])],
            axis=2
        )
        print(f"✓ Skull stripped")
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        return
    
    print("\nExtracting axial slices...")
    try:
        slices = extract_axial_slices(volume)
        print(f"✓ Extracted {len(slices)} axial slices")
    except Exception as e:
        print(f"✗ Failed to extract slices: {e}")
        return
    
    print("\nLoading model...")
    try:
        model, device = load_model()
        print(f"✓ Model loaded on device: {device}")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    print("\nRunning predictions on all slices...")
    try:
        predictions, probabilities = predict_slices_batch(model, device, slices)
        print(f"✓ Got predictions for {len(predictions)} slices")
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        return
    
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    
    normal_count = sum(1 for p in predictions if p == 0)
    tumor_count = sum(1 for p in predictions if p == 1)
    
    print(f"\nSlice-level predictions:")
    print(f"  Normal (class 0): {normal_count}/{len(predictions)} slices ({100*normal_count/len(predictions):.1f}%)")
    print(f"  Tumor (class 1):  {tumor_count}/{len(predictions)} slices ({100*tumor_count/len(predictions):.1f}%)")
    
    tumor_probs = [p[1] for p in probabilities]
    normal_probs = [p[0] for p in probabilities]
    
    print(f"\nTumor probability statistics:")
    print(f"  Mean:     {np.mean(tumor_probs):.4f}")
    print(f"  Median:   {np.median(tumor_probs):.4f}")
    print(f"  Min:      {np.min(tumor_probs):.4f}")
    print(f"  Max:      {np.max(tumor_probs):.4f}")
    print(f"  Std Dev:  {np.std(tumor_probs):.4f}")
    
    top_k = 5
    top_indices = sorted(range(len(tumor_probs)), key=lambda i: tumor_probs[i], reverse=True)[:top_k]
    
    print(f"\nTop {top_k} slices with highest tumor probability:")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i}. Slice {idx:3d}: tumor_prob={tumor_probs[idx]:.4f}, normal_prob={normal_probs[idx]:.4f}")
    
    print(f"\nPatient-level decision (using robust aggregation):")
    decision = robust_patient_prediction_from_tumor_probs(
        tumor_probs=tumor_probs,
        **load_aggregation_params(),
    )
    patient_pred = "Tumor Detected" if decision["prediction"] == 1 else "Normal"
    print(f"  Top-k score ({decision['method']}): {decision['score']:.4f}")
    print(f"  Composite risk score:         {decision['risk_score']:.4f}")
    print(f"  Suspicious slices:            {decision['suspicious_slices']}/{len(tumor_probs)}")
    print(f"  Suspicious fraction:          {decision['suspicious_fraction']:.4f}")
    print(f"  Decision: {patient_pred}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
