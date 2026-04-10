# PanCancerSeg Specialist Inference

Run one cancer-specific PanCancerSeg nnUNet model on a single CT NIfTI image and save a segmentation mask, slice PNG previews, and an MP4 overlay video.

## Model Weights

Download the trained nnUNet weights from Hugging Face: [KS987/PanCancerSeg-Specialized-weights](https://huggingface.co/KS987/PanCancerSeg-Specialized-weights)

```bash
git lfs install
git clone https://huggingface.co/KS987/PanCancerSeg-Specialized-weights
```

## Setup

Create an environment and install the Python dependencies:

```bash
pip install -r requirements.txt
```

Download the trained nnUNet model weights to a local directory. Inference resampling can require about 64 GB RAM for large 3D volumes.

Expected model layout:

```text
nnUNet_results/
|-- Dataset102_Kidney/
|   `-- nnUNetTrainerWandb2000__nnUNetResEncUNetMPlans__3d_fullres/
|       `-- fold_0/
|           `-- checkpoint_best.pth
|-- Dataset103_Liver/
|-- Dataset104_Pancreas/
`-- Dataset105_Lung/
```

## Usage

Input images can be named either `case.nii.gz` or `case_0000.nii.gz`; the script handles both.

Canonical `--cancer_type` values are `kidney_cancer`, `liver_cancer`, `pancreatic_cancer`, and `lung_cancer`.
Legacy aliases `kidney`, `liver`, `pancreas`, and `lung` are still accepted for backward compatibility.

Kidney cancer:

```bash
python predict.py --input /path/to/case.nii.gz --cancer_type kidney_cancer --model_dir /path/to/nnUNet_results --output_dir ./output
```

Liver cancer:

```bash
python predict.py --input /path/to/case.nii.gz --cancer_type liver_cancer --model_dir /path/to/nnUNet_results --output_dir ./output
```

Pancreatic cancer:

```bash
python predict.py --input /path/to/case.nii.gz --cancer_type pancreatic_cancer --model_dir /path/to/nnUNet_results --output_dir ./output
```

Lung cancer:

```bash
python predict.py --input /path/to/case.nii.gz --cancer_type lung_cancer --model_dir /path/to/nnUNet_results --output_dir ./output
```

Use CPU only when CUDA is unavailable:

```bash
python predict.py --input /path/to/case.nii.gz --cancer_type kidney_cancer --model_dir /path/to/nnUNet_results --output_dir ./output --device cpu
```

## Output Files

The output directory contains:

- `{case_id}_seg.nii.gz`: predicted segmentation mask
- `{case_id}_slice_centroid.png`: centroid slice preview
- `{case_id}_slice_max_area.png`: max predicted area slice preview
- `{case_id}_slice_extent25.png`: 25% through predicted z-extent preview
- `{case_id}_slice_extent75.png`: 75% through predicted z-extent preview
- `{case_id}_overlay.mp4`: scroll-through overlay video

## Supported Cancer Types

| Canonical `--cancer_type` | Legacy alias | Dataset | Window level | Window width |
|---------------------------|--------------|---------|-------------:|-------------:|
| kidney_cancer | kidney | Dataset102_Kidney | 40 | 400 |
| liver_cancer | liver | Dataset103_Liver | 40 | 400 |
| pancreatic_cancer | pancreas | Dataset104_Pancreas | 40 | 400 |
| lung_cancer | lung | Dataset105_Lung | -600 | 1500 |

## Troubleshooting

CUDA unavailable: run with `--device cpu` or install CUDA-enabled PyTorch.

Missing checkpoint: check that `--model_dir` points to the directory containing the `DatasetXXX_*` model folders and that `fold_0/checkpoint_best.pth` exists.

Missing custom trainer: make sure the cloned repository still has this layout:

```text
PanCancerSeg-Inference/
|-- predict.py
`-- trainers/
    `-- nnUNetTrainerWandb2000.py
```

Do not move the `trainers/` directory out of the repository. `predict.py` automatically registers the trainer with nnUNet when inference starts.

MP4 video creation failed: the segmentation mask and PNG files may still have been created. Try running the command on another machine or ask technical support to install video support for OpenCV.
