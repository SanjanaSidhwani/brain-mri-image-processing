import pickle
import gzip
from pathlib import Path
import argparse
import json
from collections import Counter

from src.dataset.dataset_builder import build_dataset_from_volumes
from src.dataset.dataset_adapter import BratsAdapter, OasisAdapter, IXIAdapter


def get_brats_volumes(brats_root):
    return BratsAdapter(brats_root).scan()


def get_oasis_volumes(oasis_root):
    return OasisAdapter(oasis_root).scan()


def get_ixi_volumes(ixi_root):
    if not ixi_root:
        return []
    root = Path(ixi_root)
    if not root.exists():
        return []
    return IXIAdapter(str(root)).scan()


def main():
    parser = argparse.ArgumentParser(description="Build serialized dataset records")
    parser.add_argument("--record_mode", type=str, default="lightweight", choices=["full", "lightweight"])
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--to_ras", action="store_true", help="Reorient to RAS before slicing")
    parser.add_argument("--target_spacing", type=float, nargs=3, default=None)
    parser.add_argument("--apply_scanner_normalization", action="store_true")
    parser.add_argument("--output", type=str, default="data/dataset_records.pkl.gz")
    parser.add_argument("--continue_on_error", action="store_true", help="Skip unreadable/corrupt volumes")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    brats20_root = project_root / "data" / "raw" / "brats2020" / "brats20-dataset-training-validation" / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"
    brats21_root = project_root / "data" / "raw" / "brats2021"
    oasis_root = project_root / "data" / "raw" / "oasis"
    ixi_root = project_root / "data" / "raw" / "IXI"

    print("Scanning datasets...")

    brats20_volumes = get_brats_volumes(str(brats20_root)) if brats20_root.exists() else []
    brats21_volumes = get_brats_volumes(str(brats21_root)) if brats21_root.exists() else []
    oasis_volumes = get_oasis_volumes(str(oasis_root))
    ixi_volumes = get_ixi_volumes(str(ixi_root))

    print(f"BraTS2020 volumes: {len(brats20_volumes)}")
    print(f"BraTS2021 volumes: {len(brats21_volumes)}")
    print(f"OASIS volumes: {len(oasis_volumes)}")
    print(f"IXI volumes: {len(ixi_volumes)}")

    if (len(brats20_volumes) + len(brats21_volumes)) == 0 or len(oasis_volumes) == 0:
        raise ValueError("Dataset scanning failed. Check dataset paths.")

    all_volumes = brats20_volumes + brats21_volumes + oasis_volumes + ixi_volumes

    print(
        "Building dataset records... "
        f"mode={args.record_mode}, to_ras={args.to_ras}, target_spacing={args.target_spacing}, "
        f"scanner_norm={args.apply_scanner_normalization}"
    )

    dataset_records = []
    skipped = []
    total_vols = len(all_volumes)

    for idx, volume_item in enumerate(all_volumes, start=1):
        try:
            recs = build_dataset_from_volumes(
                [volume_item],
                threshold=args.threshold,
                to_ras=args.to_ras,
                target_spacing=tuple(args.target_spacing) if args.target_spacing else None,
                apply_scanner_normalization=args.apply_scanner_normalization,
                record_mode=args.record_mode,
            )
            dataset_records.extend(recs)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            volume_path = getattr(volume_item, "volume_path", None)
            if volume_path is None and isinstance(volume_item, dict):
                volume_path = volume_item.get("volume_path")
            skipped.append({"volume_path": str(volume_path), "error": f"{type(exc).__name__}: {exc}"})

        if idx % 100 == 0 or idx == total_vols:
            print(
                f"Progress: {idx}/{total_vols} volumes | "
                f"records={len(dataset_records)} | skipped={len(skipped)}"
            )

    print(f"Total slice records: {len(dataset_records)}")

    modality_counts = Counter(str(r.get("modality", "missing")).lower() for r in dataset_records)
    print(f"Top modality counts: {modality_counts.most_common(12)}")

    if skipped:
        print(f"Skipped volumes: {len(skipped)}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(output_path, "wb") as f:
        pickle.dump(dataset_records, f)

    print(f"Dataset saved to: {output_path}")

    if skipped:
        skipped_path = output_path.with_name(output_path.stem + "_skipped_volumes.json")
        with open(skipped_path, "w", encoding="utf-8") as f:
            json.dump(skipped, f, indent=2)
        print(f"Skipped volume log: {skipped_path}")


if __name__ == "__main__":
    main()