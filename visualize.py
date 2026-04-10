"""Visualization helpers for single-case PanCancerSeg inference."""

from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OVERLAY_COLOR = (255, 0, 0)


def preprocess_volume(volume, modality, wl, ww):
    """Window or normalize a volume and return uint8 data in [0, 255]."""
    modality = modality.upper()
    volume = volume.astype(np.float32, copy=False)

    if modality == "CT":
        lower_bound = wl - ww / 2
        upper_bound = wl + ww / 2
        clipped = np.clip(volume, lower_bound, upper_bound)
        return _normalize_to_uint8(clipped)

    if modality != "MR":
        raise ValueError(f"Unsupported modality '{modality}'. Expected CT or MR.")

    nonzero = volume[volume != 0]
    if nonzero.size > 0:
        lower_bound, upper_bound = np.percentile(nonzero, [0.5, 99.5])
        clipped = np.clip(volume, lower_bound, upper_bound)
        normalized = _normalize_to_uint8(clipped)
        normalized[volume == 0] = 0
        return normalized

    return _normalize_to_uint8(volume)


def overlay_mask(gray_slice, mask_slice, color=DEFAULT_OVERLAY_COLOR, alpha=0.5):
    """Apply a semi-transparent RGB overlay to one grayscale slice."""
    gray_slice = np.asarray(gray_slice, dtype=np.uint8)
    if gray_slice.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale slice, got shape {gray_slice.shape}")

    rgb = np.stack([gray_slice] * 3, axis=-1)
    mask = mask_slice > 0
    if not np.any(mask):
        return rgb

    out = rgb.copy()
    color_arr = np.asarray(color, dtype=np.float32)
    blended = out[mask].astype(np.float32) * (1 - alpha) + color_arr * alpha
    out[mask] = np.clip(blended, 0, 255).astype(np.uint8)
    return out


def find_key_slices(mask_vol):
    """Return named representative z-slices for a mask in [z, y, x] order."""
    if mask_vol.ndim != 3:
        raise ValueError(f"Expected a 3D mask volume, got shape {mask_vol.shape}")

    depth = mask_vol.shape[0]
    if depth == 0:
        raise ValueError("Cannot select key slices from an empty z-dimension")

    mask = mask_vol > 0
    if np.any(mask):
        z_indices = np.where(np.any(mask, axis=(1, 2)))[0]
        areas = mask.reshape(depth, -1).sum(axis=1)
        coords = np.argwhere(mask)
        centroid_z = int(round(float(coords[:, 0].mean())))
        min_z = int(z_indices.min())
        max_z = int(z_indices.max())
        return {
            "centroid": _clip_slice(centroid_z, depth),
            "max_area": int(areas.argmax()),
            "extent25": _clip_slice(round(min_z + 0.25 * (max_z - min_z)), depth),
            "extent75": _clip_slice(round(min_z + 0.75 * (max_z - min_z)), depth),
        }

    middle = depth // 2
    offset = max(1, depth // 10)
    return {
        "centroid": middle,
        "max_area": _clip_slice(middle - offset, depth),
        "extent25": _clip_slice(middle + offset, depth),
        "extent75": _clip_slice(middle + 2 * offset, depth),
    }


def generate_slice_images(
    image_uint8,
    mask_vol,
    output_dir,
    case_name,
    color=DEFAULT_OVERLAY_COLOR,
    alpha=0.5,
):
    """Save side-by-side PNGs for representative slices."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    key_slices = find_key_slices(mask_vol)
    outputs = {}

    for label, z_idx in key_slices.items():
        gray_slice = image_uint8[z_idx]
        mask_slice = mask_vol[z_idx] > 0
        overlay = overlay_mask(gray_slice, mask_slice, color=color, alpha=alpha)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=150)
        axes[0].imshow(gray_slice, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Image")
        axes[0].axis("off")
        axes[1].imshow(overlay)
        axes[1].set_title("Segmentation overlay")
        axes[1].axis("off")
        fig.suptitle(f"{case_name} | z={z_idx}")
        fig.tight_layout()

        out_path = output_dir / f"{case_name}_slice_{label}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        outputs[label] = out_path

    return outputs


def generate_video(
    image_uint8,
    mask_vol,
    output_dir,
    case_name,
    cancer_type,
    color=DEFAULT_OVERLAY_COLOR,
    alpha=0.5,
    fps=10,
):
    """Generate an MP4 scroll-through overlay video."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{case_name}_overlay.mp4"

    start_z, end_z = _video_z_range(mask_vol)
    first_frame = _make_video_frame(
        image_uint8[start_z],
        mask_vol[start_z],
        color,
        alpha,
        start_z,
        image_uint8.shape[0],
        cancer_type,
    )
    height, width = first_frame.shape[:2]

    writer = _open_video_writer(video_path, fps, width, height)
    # Frame annotations are drawn in RGB space; convert only when writing to OpenCV.
    writer.write(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))

    for z_idx in range(start_z + 1, end_z + 1):
        frame = _make_video_frame(
            image_uint8[z_idx],
            mask_vol[z_idx],
            color,
            alpha,
            z_idx,
            image_uint8.shape[0],
            cancer_type,
        )
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    return video_path


def generate_outputs(
    image_path,
    mask_path,
    output_dir,
    case_name,
    cancer_type,
    modality,
    wl,
    ww,
    color=DEFAULT_OVERLAY_COLOR,
    alpha=0.5,
    fps=10,
):
    """Read image and mask volumes, then write PNG previews and MP4 video."""
    image = sitk.ReadImage(str(image_path))
    mask = sitk.ReadImage(str(mask_path))
    image_vol = sitk.GetArrayFromImage(image)
    mask_vol = sitk.GetArrayFromImage(mask)

    if image_vol.shape != mask_vol.shape:
        raise ValueError(
            "Image and segmentation shapes do not match: "
            f"image={image_vol.shape}, segmentation={mask_vol.shape}. "
            "Both arrays are expected in [z, y, x] order."
        )

    image_uint8 = preprocess_volume(image_vol, modality, wl, ww)
    slice_paths = generate_slice_images(
        image_uint8,
        mask_vol,
        output_dir,
        case_name,
        color,
        alpha,
    )
    video_path = generate_video(
        image_uint8,
        mask_vol,
        output_dir,
        case_name,
        cancer_type,
        color,
        alpha,
        fps,
    )
    return {"slices": slice_paths, "video": video_path}


def _normalize_to_uint8(volume):
    v_min = float(np.min(volume))
    v_max = float(np.max(volume))
    if not np.isfinite(v_min) or not np.isfinite(v_max) or v_max <= v_min:
        return np.zeros(volume.shape, dtype=np.uint8)
    normalized = (volume - v_min) / (v_max - v_min) * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def _clip_slice(index, depth):
    return int(np.clip(index, 0, depth - 1))


def _video_z_range(mask_vol, padding=10, empty_window=80):
    depth = mask_vol.shape[0]
    mask = mask_vol > 0
    if np.any(mask):
        z_indices = np.where(np.any(mask, axis=(1, 2)))[0]
        return (
            max(0, int(z_indices.min()) - padding),
            min(depth - 1, int(z_indices.max()) + padding),
        )

    if depth <= empty_window:
        return 0, depth - 1
    middle = depth // 2
    half = empty_window // 2
    return max(0, middle - half), min(depth - 1, middle + half)


def _make_video_frame(gray_slice, mask_slice, color, alpha, z_idx, depth, cancer_type):
    frame = overlay_mask(gray_slice, mask_slice, color=color, alpha=alpha)
    frame = _upscale_if_small(frame)

    annotation = f"Slice {z_idx + 1}/{depth} | {cancer_type}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, min(frame.shape[:2]) / 900)
    thickness = max(1, int(round(font_scale * 2)))
    text_size, baseline = cv2.getTextSize(annotation, font, font_scale, thickness)
    x, y = 12, 12 + text_size[1]
    cv2.rectangle(
        frame,
        (x - 6, y - text_size[1] - 6),
        (x + text_size[0] + 6, y + baseline + 6),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.putText(frame, annotation, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def _upscale_if_small(frame, min_short_side=512):
    height, width = frame.shape[:2]
    short_side = min(height, width)
    if short_side >= min_short_side:
        return frame
    scale = min_short_side / short_side
    new_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_LINEAR)


def _open_video_writer(video_path, fps, width, height):
    attempts = [
        ("avc1", "H.264/avc1"),
        ("mp4v", "MPEG-4/mp4v"),
    ]
    for fourcc_text, label in attempts:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_text)
        writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), (width, height))
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(
        f"Could not open MP4 writer at {video_path}. Tried "
        + ", ".join(label for _, label in attempts)
        + ". Install an OpenCV build with MP4 codec support or try another machine."
    )
