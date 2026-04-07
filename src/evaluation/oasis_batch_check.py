import argparse
from pathlib import Path

import numpy as np
import torch

from src.aggregation.topk_aggregation import robust_patient_prediction_from_tumor_probs
from src.dataset.input_transforms import build_eval_transform
from src.inference import get_latest_checkpoint, load_aggregation_params
from src.models.model_factory import create_model
from src.preprocessing.slice_utils import extract_axial_slices
from src.preprocessing.volume_utils import load_nifti, zscore_normalize, strip_skull


def preprocess_slice(slice_2d, transform):
    arr = np.asarray(slice_2d, dtype=np.float32)
    stacked = np.stack([arr, arr, arr], axis=0)
    tensor = torch.from_numpy(stacked).float()
    return transform(tensor)


def collect_oasis_files(root: Path):
    files = list(root.rglob("*.nii")) + list(root.rglob("*.nii.gz"))
    files = sorted(set(files))
    return files


def main():
    default_oasis_root = str(Path(__file__).resolve().parents[2] / "data" / "raw" / "oasis")

    parser = argparse.ArgumentParser(description="Batch OASIS false-positive check")
    parser.add_argument(
        "--oasis_root",
        type=str,
        default=default_oasis_root,
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_cases", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint or get_latest_checkpoint("outputs/checkpoints")
    params = load_aggregation_params()

    model = create_model(architecture="cnn", num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    transform = build_eval_transform(target_size=224, center_crop_size=180)

    oasis_root = Path(args.oasis_root)
    files = collect_oasis_files(oasis_root)
    if args.max_cases and args.max_cases > 0:
        files = files[: args.max_cases]

    total = len(files)
    if total == 0:
        raise FileNotFoundError(f"No OASIS files found under {oasis_root}")

    fp_count = 0
    normal_count = 0
    failed = 0
    fp_examples = []

    print(f"Using device: {device}")
    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Using aggregation params: {params}")
    print(f"Evaluating OASIS cases: {total}")

    with torch.no_grad():
        for idx, file_path in enumerate(files, start=1):
            try:
                volume = load_nifti(str(file_path))
                volume = zscore_normalize(volume)
                volume = np.stack(
                    [strip_skull(volume[:, :, i]) for i in range(volume.shape[2])],
                    axis=2,
                )
                slices = extract_axial_slices(volume)

                tumor_probs = []
                for s in slices:
                    slice_tensor = preprocess_slice(s, transform).unsqueeze(0).to(device)
                    out = model(slice_tensor)
                    probs = torch.softmax(out, dim=1)
                    tumor_probs.append(float(probs[0, 1].item()))

                decision = robust_patient_prediction_from_tumor_probs(
                    tumor_probs=tumor_probs,
                    threshold=params["threshold"],
                    top_k=params["top_k"],
                    method=params["method"],
                    min_suspicious_slices=params["min_suspicious_slices"],
                    suspicious_prob_threshold=params["suspicious_prob_threshold"],
                    min_suspicious_fraction=params["min_suspicious_fraction"],
                )

                if decision["prediction"] == 1:
                    fp_count += 1
                    fp_examples.append(
                        {
                            "file": file_path.name,
                            "risk_score": round(float(decision["risk_score"]), 6),
                            "topk_score": round(float(decision["score"]), 6),
                            "suspicious_fraction": round(float(decision["suspicious_fraction"]), 6),
                            "suspicious_slices": int(decision["suspicious_slices"]),
                        }
                    )
                else:
                    normal_count += 1

            except Exception:
                failed += 1

            if idx % 25 == 0 or idx == total:
                print(f"Progress {idx}/{total} | FP: {fp_count} | Normal: {normal_count} | Failed: {failed}")

    evaluated = total - failed
    fpr = (fp_count / evaluated) if evaluated > 0 else 0.0
    specificity = 1.0 - fpr

    print("\n===== OASIS Batch Summary =====")
    print(f"Evaluated: {evaluated}")
    print(f"False Positives: {fp_count}")
    print(f"True Negatives: {normal_count}")
    print(f"Failed: {failed}")
    print(f"False Positive Rate: {fpr:.4f}")
    print(f"Specificity: {specificity:.4f}")

    if fp_examples:
        fp_examples = sorted(fp_examples, key=lambda x: x["risk_score"], reverse=True)
        print("\nTop false-positive examples:")
        for item in fp_examples[:10]:
            print(item)


if __name__ == "__main__":
    main()
