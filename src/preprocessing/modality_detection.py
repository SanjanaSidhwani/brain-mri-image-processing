from __future__ import annotations

from pathlib import Path
from typing import Optional

import nibabel as nib


MODALITY_ALIASES = {
    "flair": "flair",
    "t2flair": "flair",
    "t1": "t1",
    "t1w": "t1",
    "t1c": "t1c",
    "t1ce": "t1c",
    "t1gd": "t1c",
    "t1post": "t1c",
    "postcontrast": "t1c",
    "t2": "t2",
    "t2w": "t2",
    "pd": "pd",
    "protondensity": "pd",
}

MODALITY_STOPWORDS = {
    "brats",
    "training",
    "validation",
    "patient",
    "subject",
    "scan",
    "series",
    "image",
    "volume",
    "nii",
    "gz",
    "mr",
    "mri",
    "anon",
    "masked",
    "brain",
    "oasis",
    "ixi",
}

CUSTOM_MODALITY_HINTS = {
    "adc",
    "dwi",
    "dti",
    "swi",
    "gre",
    "epi",
    "asl",
    "tof",
    "mprage",
    "mpr",
    "mra",
    "fse",
    "b0",
    "b1000",
    "tracew",
}


def _tokenized(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in text)


def _extract_header_text(img: nib.spatialimages.SpatialImage) -> str:
    header = img.header
    bits = []
    for key in ("descrip", "aux_file", "db_name"):
        if key in header:
            raw = header[key]
            if isinstance(raw, bytes):
                bits.append(raw.decode("utf-8", errors="ignore"))
            else:
                bits.append(str(raw))
    return " ".join(bits)


def _infer_custom_modality(tokens) -> str:
    for token in tokens:
        if token in CUSTOM_MODALITY_HINTS:
            return token

    for token in tokens:
        if token in MODALITY_STOPWORDS:
            continue

        # Keep concise alpha-numeric modality-like tokens.
        if 2 <= len(token) <= 16 and any(ch.isalpha() for ch in token):
            if token.startswith("t1") or token.startswith("t2"):
                continue
            if token.isdigit():
                continue
            return token

    return "unknown"


def detect_modality(file_path: str, img: Optional[nib.spatialimages.SpatialImage] = None) -> str:
    candidate = Path(file_path).name
    text = _tokenized(candidate)

    if img is not None:
        text = f"{text} {_tokenized(_extract_header_text(img))}"

    tokens = text.split()

    # Prefer explicit T1-contrast tokens before base T1 token.
    for token in tokens:
        if token in ("t1ce", "t1c", "t1gd", "t1post", "postcontrast"):
            return "t1c"

    for token in tokens:
        if token in MODALITY_ALIASES:
            return MODALITY_ALIASES[token]

    if "flair" in text:
        return "flair"
    if "t1ce" in text or "t1c" in text:
        return "t1c"
    if "t2" in text:
        return "t2"
    if "t1" in text:
        return "t1"
    if "pd" in text or "proton density" in text:
        return "pd"

    return _infer_custom_modality(tokens)


def detect_field_strength_t(file_path: str, img: Optional[nib.spatialimages.SpatialImage] = None) -> Optional[float]:
    candidate = _tokenized(Path(file_path).name)

    if img is not None:
        candidate = f"{candidate} {_tokenized(_extract_header_text(img))}"

    if "3t" in candidate or "3 0t" in candidate or "3.0t" in candidate:
        return 3.0
    if "15t" in candidate or "1 5t" in candidate or "1.5t" in candidate:
        return 1.5

    return None
