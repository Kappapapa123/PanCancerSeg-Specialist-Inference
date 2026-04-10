"""Run single-case PanCancerSeg nnUNet inference and visualization."""

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

from visualize import generate_outputs


CANCER_CONFIGS = {
    "kidney": {
        "dataset_id": 102,
        "dataset_name": "Dataset102_Kidney",
        "wl": 40,
        "ww": 400,
        "color": (255, 0, 0),
    },
    "liver": {
        "dataset_id": 103,
        "dataset_name": "Dataset103_Liver",
        "wl": 40,
        "ww": 400,
        "color": (255, 0, 0),
    },
    "pancreas": {
        "dataset_id": 104,
        "dataset_name": "Dataset104_Pancreas",
        "wl": 40,
        "ww": 400,
        "color": (255, 0, 0),
    },
    "lung": {
        "dataset_id": 105,
        "dataset_name": "Dataset105_Lung",
        "wl": -600,
        "ww": 1500,
        "color": (255, 0, 0),
    },
}

TRAINER_NAME = "nnUNetTrainerWandb2000"
PLANS_NAME = "nnUNetResEncUNetMPlans"
CONFIGURATION = "3d_fullres"
CHECKPOINT_NAME = "checkpoint_best.pth"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one PanCancerSeg cancer-specific nnUNet model on a single NIfTI image."
    )
    parser.add_argument("--input", required=True, help="Path to a single .nii.gz image")
    parser.add_argument(
        "--cancer_type",
        required=True,
        choices=sorted(CANCER_CONFIGS),
        help="Cancer-specific model to use",
    )
    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to nnUNet results directory containing DatasetXXX_* folders",
    )
    parser.add_argument("--output_dir", default="./output", help="Where to save results")
    parser.add_argument("--modality", choices=["CT", "MR"], default="CT")
    parser.add_argument("--fps", type=int, default=10, help="Video frames per second")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")
    if input_path.name.startswith("._") or not input_path.name.endswith(".nii.gz"):
        raise ValueError(f"Expected a .nii.gz image, got: {input_path.name}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Use --device cpu or install CUDA-enabled PyTorch."
        )
    if args.fps <= 0:
        raise ValueError("--fps must be a positive integer")

    output_dir.mkdir(parents=True, exist_ok=True)
    config = CANCER_CONFIGS[args.cancer_type]
    case_id = resolve_case_id(input_path)

    install_custom_trainer()
    model_folder = resolve_model_folder(model_dir, config["dataset_name"])

    with tempfile.TemporaryDirectory(prefix="pancancerseg_") as tmp:
        tmp_path = Path(tmp)
        tmp_input_dir = tmp_path / "input"
        tmp_output_dir = tmp_path / "prediction"
        tmp_input_dir.mkdir()
        tmp_output_dir.mkdir()

        nnunet_input = tmp_input_dir / f"{case_id}_0000.nii.gz"
        symlink_or_copy(input_path, nnunet_input)

        run_nnunet_prediction(
            model_folder=model_folder,
            input_dir=tmp_input_dir,
            output_dir=tmp_output_dir,
            device=args.device,
        )

        raw_seg = tmp_output_dir / f"{case_id}.nii.gz"
        if not raw_seg.exists():
            produced = sorted(tmp_output_dir.glob("*.nii.gz"))
            raise FileNotFoundError(
                f"nnUNet did not write the expected segmentation {raw_seg}. "
                f"Found: {[p.name for p in produced]}"
            )

        seg_path = output_dir / f"{case_id}_seg.nii.gz"
        shutil.copy2(raw_seg, seg_path)

    viz_outputs = generate_outputs(
        image_path=input_path,
        mask_path=seg_path,
        output_dir=output_dir,
        case_name=case_id,
        cancer_type=args.cancer_type,
        modality=args.modality,
        wl=config["wl"],
        ww=config["ww"],
        color=config["color"],
        alpha=0.5,
        fps=args.fps,
    )

    positive_voxels, tumor_volume_ml = summarize_segmentation(seg_path)
    print_summary(seg_path, viz_outputs, positive_voxels, tumor_volume_ml)


def resolve_case_id(input_path):
    name = input_path.name
    if not name.endswith(".nii.gz"):
        raise ValueError(f"Expected a .nii.gz image, got: {name}")
    case_id = name[: -len(".nii.gz")]
    if case_id.endswith("_0000"):
        case_id = case_id[: -len("_0000")]
    if not case_id:
        raise ValueError(f"Could not resolve a case ID from: {input_path}")
    return case_id


def install_custom_trainer():
    import nnunetv2

    src = Path(__file__).resolve().parent / "trainers" / f"{TRAINER_NAME}.py"
    if not src.exists():
        raise FileNotFoundError(f"Custom trainer file is missing: {src}")

    variants_dir = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer" / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    dst = variants_dir / src.name

    if dst.exists() or dst.is_symlink():
        try:
            if dst.resolve() == src.resolve():
                return dst
        except OSError:
            pass
        dst.unlink()

    try:
        dst.symlink_to(src.resolve())
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)
    print(f"Installed custom trainer: {dst}")
    return dst


def resolve_model_folder(model_dir, dataset_name):
    model_folder = (
        model_dir
        / dataset_name
        / f"{TRAINER_NAME}__{PLANS_NAME}__{CONFIGURATION}"
    )
    checkpoint = model_folder / "fold_0" / CHECKPOINT_NAME
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Expected checkpoint not found: {checkpoint}. "
            "Check --model_dir and make sure the trained weights are downloaded."
        )
    return model_folder


def symlink_or_copy(src, dst):
    try:
        dst.symlink_to(src.resolve())
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def run_nnunet_prediction(model_folder, input_dir, output_dir, device):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=(device == "cuda"),
        device=torch.device(device),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=(0,),
        checkpoint_name=CHECKPOINT_NAME,
    )
    predictor.predict_from_files(
        str(input_dir),
        str(output_dir),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )


def summarize_segmentation(seg_path):
    seg = sitk.ReadImage(str(seg_path))
    seg_arr = sitk.GetArrayFromImage(seg)
    positive_voxels = int(np.count_nonzero(seg_arr))
    spacing_x, spacing_y, spacing_z = seg.GetSpacing()
    tumor_volume_ml = positive_voxels * spacing_x * spacing_y * spacing_z / 1000.0
    return positive_voxels, tumor_volume_ml


def print_summary(seg_path, viz_outputs, positive_voxels, tumor_volume_ml):
    print("\nPanCancerSeg inference complete")
    print(f"Segmentation mask : {seg_path}")
    print("Slice PNGs        :")
    for label, path in viz_outputs["slices"].items():
        print(f"  {label:9s} : {path}")
    print(f"Overlay video     : {viz_outputs['video']}")
    print(f"Positive voxels   : {positive_voxels}")
    print(f"Tumor volume      : {tumor_volume_ml:.3f} mL")


if __name__ == "__main__":
    main()
