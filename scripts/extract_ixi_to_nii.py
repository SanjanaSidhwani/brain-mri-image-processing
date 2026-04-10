import argparse
import gzip
import shutil
import tarfile
from pathlib import Path


def safe_extract_tar(archive_path: Path, target_dir: Path) -> int:
    extracted = 0
    with tarfile.open(archive_path, "r:*") as tf:
        for member in tf.getmembers():
            member_path = target_dir / member.name
            resolved_target = target_dir.resolve()
            resolved_member = member_path.resolve()

            # Prevent path traversal on extraction.
            if not str(resolved_member).startswith(str(resolved_target)):
                raise RuntimeError(f"Unsafe path in archive {archive_path}: {member.name}")

            tf.extract(member, path=target_dir)
            if member.isfile():
                extracted += 1
    return extracted


def gunzip_to_nii(gz_path: Path, keep_gz: bool) -> Path:
    if gz_path.suffix != ".gz" or not gz_path.name.endswith(".nii.gz"):
        raise ValueError(f"Not a .nii.gz file: {gz_path}")

    nii_path = gz_path.with_suffix("")
    with gzip.open(gz_path, "rb") as src, open(nii_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

    if not keep_gz:
        gz_path.unlink()

    return nii_path


def extract_ixi_archives(
    ixi_dir: Path,
    output_dir: Path,
    keep_nii_gz: bool,
    skip_existing_nii: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    archives = sorted(
        [p for p in ixi_dir.glob("*.tar") if p.is_file()]
        + [p for p in ixi_dir.glob("*.tar.gz") if p.is_file()]
        + [p for p in ixi_dir.glob("*.tgz") if p.is_file()]
    )

    if not archives:
        print(f"No IXI archives found in: {ixi_dir}")
        return

    print(f"Found {len(archives)} archive(s).")

    total_files_extracted = 0
    for archive in archives:
        print(f"Extracting: {archive.name}")
        count = safe_extract_tar(archive, output_dir)
        total_files_extracted += count

    print(f"Extracted {total_files_extracted} file(s) from archives.")

    gz_files = sorted(output_dir.rglob("*.nii.gz"))
    print(f"Found {len(gz_files)} .nii.gz file(s) to convert.")

    converted = 0
    skipped = 0
    for gz_file in gz_files:
        nii_file = gz_file.with_suffix("")
        if skip_existing_nii and nii_file.exists():
            skipped += 1
            if not keep_nii_gz:
                gz_file.unlink()
            continue

        gunzip_to_nii(gz_file, keep_gz=keep_nii_gz)
        converted += 1

    print(f"Converted {converted} file(s) to .nii.")
    if skipped:
        print(f"Skipped {skipped} existing .nii file(s).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract IXI archives and convert extracted .nii.gz files to .nii"
    )
    parser.add_argument(
        "--ixi_dir",
        type=str,
        default="data/raw/IXI",
        help="Directory containing IXI archives (.tar/.tar.gz/.tgz)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw/IXI",
        help="Extraction destination directory",
    )
    parser.add_argument(
        "--keep_nii_gz",
        action="store_true",
        help="Keep .nii.gz files after creating .nii",
    )
    parser.add_argument(
        "--no_skip_existing_nii",
        action="store_true",
        help="Recreate .nii files even if they already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ixi_dir = Path(args.ixi_dir)
    output_dir = Path(args.output_dir)

    if not ixi_dir.exists():
        raise FileNotFoundError(f"IXI directory not found: {ixi_dir}")

    extract_ixi_archives(
        ixi_dir=ixi_dir,
        output_dir=output_dir,
        keep_nii_gz=bool(args.keep_nii_gz),
        skip_existing_nii=not bool(args.no_skip_existing_nii),
    )


if __name__ == "__main__":
    main()
