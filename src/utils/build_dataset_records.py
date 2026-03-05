import pickle
import gzip
from pathlib import Path

from src.dataset.dataset_builder import build_dataset_from_volumes


def main():

    sample_volumes = [
        ("path/to/brats_volume1.nii.gz", 1, "patient_001", "brats"),
        ("path/to/oasis_volume1.nii.gz", 0, "patient_002", "oasis"),
    ]

    print("Building dataset records...")

    dataset_records = build_dataset_from_volumes(sample_volumes)

    print(f"Total slice records: {len(dataset_records)}")

    output_path = Path("data/dataset_records.pkl.gz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(output_path, "wb") as f:
        pickle.dump(dataset_records, f)

    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    main()