"""Training image decoding into a shared, preallocated uint8 image buffer.

Keep decoding and conversion inside the caller's measured interval. Importing
this module does not import torch or cv2.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


def _pil_rgb(path, shape_hw):
    from PIL import Image

    h, w = shape_hw
    with Image.open(path) as image:
        return np.asarray(
            image.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
        )


def _png_rgb8_or_pil(path, shape_hw, *, decoder_report=None):
    """Decode RGB8 PNG with OpenCV; retain Pillow for every other input.

    Eligibility comes from the PNG signature and IHDR, never a PNG suffix.
    JPEG suffixes go straight to Pillow without an extra header read. Alpha,
    palette, grayscale and higher bit depths retain Pillow's RGB conversion.
    No EXIF transpose, gamma, ICC or other pixel transform is introduced.

    An optional caller-owned dict receives the actual ``decoder`` (``pillow``
    or ``rgb8_png``) and ``fallback_reason`` after successful decoding.
    """
    from PIL import Image

    if decoder_report is not None:
        decoder_report.clear()

    def pillow(reason):
        rgb = _pil_rgb(path, shape_hw)
        if decoder_report is not None:
            decoder_report.update(decoder="pillow", fallback_reason=reason)
        return rgb

    if Path(path).suffix.lower() in {".jpg", ".jpeg", ".jpe", ".jfif"}:
        return pillow("jpeg_suffix")

    try:
        with open(path, "rb") as stream:
            header = stream.read(29)
    except OSError:
        # Preserve Pillow's original failure behavior for unreadable files.
        return pillow("header_read_error")

    eligible = (
        len(header) == 29
        and header[:8] == b"\x89PNG\r\n\x1a\n"
        and header[8:16] == b"\x00\x00\x00\rIHDR"
        and header[24:26] == b"\x08\x02"
        and header[26:28] == b"\x00\x00"
        and header[28] in (0, 1)
    )
    if not eligible:
        return pillow("not_rgb8_png")

    try:
        import cv2
    except (ImportError, OSError):
        # cv2 is optional, including when its native library cannot be loaded.
        return pillow("opencv_unavailable")

    try:
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if (
            array is None
            or array.dtype != np.uint8
            or array.ndim != 3
            or array.shape[2] != 3
        ):
            # This also keeps RGB PNG with transparency on the Pillow path.
            return pillow("opencv_output_not_rgb8")
        rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    except cv2.error:
        return pillow("opencv_decode_error")

    h, w = shape_hw
    result = np.asarray(
        Image.fromarray(rgb).resize((w, h), Image.Resampling.LANCZOS)
    )
    if decoder_report is not None:
        decoder_report.update(decoder="rgb8_png", fallback_reason=None)
    return result


def load_training_images(
    paths, indices, shape_hw, workers=16, *, decoder_report=None
):
    """Load selected images with RGB8 PNG decoding and Pillow fallback.

    Workers fill distinct slots of one NHWC uint8 allocation, preserving order
    and duplicate indices. The validated permute/float/div_ sequence preserves
    float32 NCHW values and strides; no contiguous conversion or GPU transfer
    is added.

    Pass a fresh dict as ``decoder_report`` to query what this call actually
    used. On success it contains requested_decoder (always rgb8_png),
    actual_decoder (pillow/rgb8_png/mixed), decoder_counts, fallback_counts and
    per-slot images in indices order. Its status remains incomplete if loading
    raises. Reports belong to the caller, with no shared mutable state.
    """
    import torch

    if decoder_report is not None:
        decoder_report.clear()
        decoder_report.update(
            status="incomplete",
            requested_decoder="rgb8_png",
        )

    indices = list(indices)
    if not indices:
        # Preserve the original np.stack error for an empty selection.
        raise ValueError("need at least one array to stack")
    h, w = shape_hw
    pixels = np.empty((len(indices), h, w, 3), dtype=np.uint8)
    reports = [{} for _ in indices] if decoder_report is not None else None

    def fill(slot):
        report = reports[slot] if reports is not None else None
        pixels[slot] = _png_rgb8_or_pil(
            paths[indices[slot]], shape_hw, decoder_report=report
        )

    if workers == 1:
        for slot in range(len(indices)):
            fill(slot)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for _ in pool.map(fill, range(len(indices))):
                pass

    result = torch.from_numpy(pixels).permute(0, 3, 1, 2).float().div_(255)
    if decoder_report is not None:
        counts = {"pillow": 0, "rgb8_png": 0}
        fallbacks = {}
        for report in reports:
            counts[report["decoder"]] += 1
            reason = report["fallback_reason"]
            if reason is not None:
                fallbacks[reason] = fallbacks.get(reason, 0) + 1
        used = [name for name, count in counts.items() if count]
        decoder_report.update(
            status="complete",
            actual_decoder=used[0] if len(used) == 1 else "mixed",
            decoder_counts=counts,
            fallback_counts=fallbacks,
            images=reports,
        )
    return result
