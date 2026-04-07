import pickle
import gzip
from pathlib import Path

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
    project_root = Path(__file__).resolve().parents[2]
    brats_root = project_root / "data" / "raw" / "brats"
    oasis_root = project_root / "data" / "raw" / "oasis"
    ixi_root = project_root / "data" / "raw" / "ixi"

    print("Scanning datasets...")

    brats_volumes = get_brats_volumes(str(brats_root))
    oasis_volumes = get_oasis_volumes(str(oasis_root))
    ixi_volumes = get_ixi_volumes(str(ixi_root))

    print(f"BraTS volumes: {len(brats_volumes)}")
    print(f"OASIS volumes: {len(oasis_volumes)}")
    print(f"IXI volumes: {len(ixi_volumes)}")

    if len(brats_volumes) == 0 or len(oasis_volumes) == 0:
        raise ValueError("Dataset scanning failed. Check dataset paths.")

    all_volumes = brats_volumes + oasis_volumes + ixi_volumes

    print("Building dataset records... (this will take time)")

    dataset_records = build_dataset_from_volumes(all_volumes)

    print(f"Total slice records: {len(dataset_records)}")

    output_path = Path("data/dataset_records.pkl.gz")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(output_path, "wb") as f:
        pickle.dump(dataset_records, f)

    print(f"Dataset saved to: {output_path}")


if __name__ == "__main__":
    main()