# PanCancerSeg Inference

Run one cancer-specific PanCancerSeg nnUNet model on a single CT or MR NIfTI image and save a segmentation mask, slice PNG previews, and an MP4 overlay video.

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

Kidney:

```bash
python predict.py --input /path/to/case.nii.gz --cancer_type kidney --model_dir /path/to/nnUNet_results --output_dir ./output
```

Liver:

```bash
python predict.py --input /path/to/case.nii.gz --cancer_type liver --model_dir /path/to/nnUNet_results --output_dir ./output
```

Pancreas:

```bash
python predict.py --input /path/to/case.nii.gz --cancer_type pancreas --model_dir /path/to/nnUNet_results --output_dir ./output
```

Lung:

```bash
python predict.py --input /path/to/case.nii.gz --cancer_type lung --model_dir /path/to/nnUNet_results --output_dir ./output
```

Use CPU only when CUDA is unavailable:

```bash
python predict.py --input /path/to/case.nii.gz --cancer_type kidney --model_dir /path/to/nnUNet_results --output_dir ./output --device cpu
```

## Output Files

The output directory contains:

- `{case_id}_seg.nii.gz`: predicted segmentation mask
- `{case_id}_slice_centroid.png`: centroid slice preview
- `{case_id}_slice_max_area.png`: max predicted area slice preview
- `{case_id}_slice_extent25.png`: 25% through predicted z-extent preview
- `{case_id}_slice_extent75.png`: 75% through predicted z-extent preview
- `{case_id}_overlay.mp4`: scroll-through overlay video

## CT vs MR

CT is the default:

```bash
python predict.py --input /path/to/case.nii.gz --cancer_type liver --model_dir /path/to/nnUNet_results --output_dir ./output --modality CT
```

For MR images, use:

```bash
python predict.py --input /path/to/case.nii.gz --cancer_type liver --model_dir /path/to/nnUNet_results --output_dir ./output --modality MR
```

MR visualization uses percentile-based naive clipping on nonzero voxels.

## Supported Cancer Types

| Cancer | Dataset | Window level | Window width |
|--------|---------|-------------:|-------------:|
| kidney | Dataset102_Kidney | 40 | 400 |
| liver | Dataset103_Liver | 40 | 400 |
| pancreas | Dataset104_Pancreas | 40 | 400 |
| lung | Dataset105_Lung | -600 | 1500 |

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
