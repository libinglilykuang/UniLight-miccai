# Uni-Light

Official PyTorch implementation of **“Uni-Light: An Ultra-Lightweight Framework via Uncertainty-Aware Knowledge Distillation for Brain Tumour Segmentation”**, accepted at the MICCAI 2026 DEMI Workshop.

Uni-Light is an ultra-lightweight framework for 3D brain tumour segmentation. It combines:

* a compact multi-scale student architecture;
* Dynamic Weighted Knowledge Distillation (DWKD);
* deep supervision;
* Reverse Cross-Entropy (RCE) loss;
* an SDF-based boundary loss.

The training pipeline consists of two stages:

1. Train a high-capacity Residual 3D U-Net teacher implemented using MONAI.
2. Distil knowledge from the trained teacher into the lightweight Uni-Light student model.

The experiments are conducted on the BraTS2023-GLI and MSD-BTS datasets using five-fold cross-validation.

## Project Structure

```text
UniLight-miccai/
├── data/                  # Dataset loading and preprocessing
├── models/                # Teacher and student model definitions
├── utils/                 # Loss functions, metrics, and utilities
├── train_teacher.py       # Residual 3D U-Net teacher training
├── train.py               # Uni-Light student distillation
├── requirements.txt       # Python dependencies
├── LICENSE                # Software licence
└── README.md
```

## Requirements

The code has been tested using:

* Python 3.10
* PyTorch 2.5.1
* CUDA 12.1
* MONAI

A CUDA-compatible NVIDIA GPU is recommended.

## Installation

Clone the repository:

```bash
git clone https://github.com/libinglilykuang/UniLight-miccai.git
cd UniLight-miccai
```

Create and activate a Conda environment:

```bash
conda create -n unilight python=3.10 -y
conda activate unilight
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

### BraTS2023-GLI

The BraTS2023-GLI dataset contains multi-modal MRI scans and tumour segmentation annotations.

Each case contains four MRI modalities:

* T1
* T1ce
* T2
* FLAIR

The expected directory structure is:

```text
bratsglioma/
└── Training/
    ├── BraTS-GLI-00000-000/
    │   ├── BraTS-GLI-00000-000-t1n.nii.gz
    │   ├── BraTS-GLI-00000-000-t1c.nii.gz
    │   ├── BraTS-GLI-00000-000-t2w.nii.gz
    │   ├── BraTS-GLI-00000-000-t2f.nii.gz
    │   └── BraTS-GLI-00000-000-seg.nii.gz
    └── ...
```

The dataset path must be configured before training:

```python
data_dir = "/path/to/bratsglioma/Training"
```

### MSD Brain Tumour Dataset

The MSD Brain Tumour dataset uses the following structure:

```text
Task01_BrainTumour/
├── imagesTr/
├── labelsTr/
├── imagesTs/
└── dataset.json
```

Configure the dataset path as:

```python
data_dir = "/path/to/Task01_BrainTumour"
```

The MSD-BTS preprocessing and loader configuration should be selected before training.

## Stage 1: Train the Teacher Model

The teacher is a high-capacity Residual 3D U-Net implemented using MONAI.

Configure the dataset path, output directory, and training parameters in `train_teacher.py`, then run:

```bash
python -u train_teacher.py
```

The best teacher checkpoint is saved and subsequently used for student knowledge distillation.

To resume teacher training, configure the checkpoint path:

```python
resume_path = "teacher_results/last_teacher.pth"
```

Then run:

```bash
python -u train_teacher.py
```

## Stage 2: Train Uni-Light

Before training the student, configure the teacher checkpoint in `train.py`:

```python
teacher_checkpoint = "teacher_results/best_teacher.pth"
```

Run student knowledge distillation:

```bash
python -u train.py
```

During training:

* the teacher model is loaded and frozen;
* voxel-wise teacher uncertainty is estimated using prediction entropy;
* uncertainty-weighted knowledge distillation is applied;
* the student is optimised using task, DWKD, RCE, and boundary losses;
* the checkpoint with the best validation performance is saved.

To resume student training, configure:

```python
resume_path = "student_results/last_student.pth"
```

## Cross-Validation

The experiments reported in the paper use five-fold cross-validation.

The folds should be generated using fixed seeds and must remain identical across all compared methods. Training, validation, and testing cases must not overlap within each fold.

## Evaluation

The main evaluation metrics are:

* Dice Similarity Coefficient;
* 95% Hausdorff Distance;
* number of parameters;
* FLOPs;
* inference latency;
* inference memory;
* training memory.

Add the corresponding evaluation command here when the public evaluation script is available:

```bash
python evaluate.py \
    --checkpoint student_results/best_student.pth \
    --data-dir /path/to/dataset
```

## Citation

Please cite the following paper when using this repository:

```bibtex
@inproceedings{kuang2026unilight,
  title={Uni-Light: An Ultra-Lightweight Framework via Uncertainty-Aware Knowledge Distillation for Brain Tumour Segmentation},
  author={Kuang, Libing and Salehi, Soren and Wu, Ziling and Tafti, Ahmad P. and Moemeni, Armaghan},
  booktitle={MICCAI 2026 DEMI Workshop},
  year={2026}
}
```

The official publication DOI and proceedings information will be added after publication.

## Licence

This software is made available under the PolyForm Noncommercial
License 1.0.0 for non-commercial research and educational purposes.

Commercial use is not permitted without prior written permission.
Please refer to the `LICENSE` file for the complete terms.

## Contact

For questions regarding the implementation, please open an issue in this repository.
