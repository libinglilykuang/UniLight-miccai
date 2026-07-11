# Uni-Light

Official PyTorch implementation of **Uni-Light**, an ultra-lightweight knowledge distillation framework for 3D brain tumour segmentation.

The training pipeline consists of two stages:

1. Train a **MONAI Unet teacher model**.
2. Distil the knowledge from the trained teacher model into the lightweight **Uni-Light student model**.

The dataset is divided into fixed training, validation, and testing subsets using a 5-fold cross validation.

## Project Structure

```text
UniLight/
├── data/                  # Dataset loading and preprocessing
├── models/                # Teacher and student model definitions
├── utils/                 # Loss functions, metrics, and utilities
├── train_teacher.py       # Unet teacher training
├── train.py               # Uni-Light student distillation
└── README.md
```

## Requirements

The project requires Python, PyTorch, CUDA, and MONAI.

The code has been tested with the following environment:

```text
Python 3.10
PyTorch 2.5.1
CUDA 12.1
MONAI
```

A CUDA-compatible NVIDIA GPU is recommended for training.

## Installation

Clone the repository:

```bash
git clone https://github.com/libinglilykuang/UniLight.git
cd UniLight
```

Create and activate a Conda environment:

```bash
conda create -n unilight python=3.10 -y
conda activate unilight
```

Install PyTorch and the required Python packages:

```bash
pip install torch torchvision torchaudio
pip install monai nibabel numpy scipy scikit-learn tqdm tensorboard
```

You can also check the GPU status with:

```bash
nvidia-smi
```

## Dataset

## Datasets

This project supports two brain tumour segmentation datasets.

### 1. BraTS Glioma Dataset

The main dataset used for training the SAXNet teacher model and the Uni-Light student model is available from Kaggle:

[Download the BraTS Glioma Dataset from Kaggle](https://www.kaggle.com/datasets/ahmedelzayat77/bratsglioma)

After downloading and extracting the dataset, the expected directory is:

```text
bratsglioma/
└── Training/
    ├── BraTS-GLI-00000-000/
    │   ├── BraTS-GLI-00000-000-t1n.nii.gz
    │   ├── BraTS-GLI-00000-000-t1c.nii.gz
    │   ├── BraTS-GLI-00000-000-t2w.nii.gz
    │   ├── BraTS-GLI-00000-000-t2f.nii.gz
    │   └── BraTS-GLI-00000-000-seg.nii.gz
    ├── BraTS-GLI-00001-000/
    └── ...
```

Set the dataset path used by the training scripts to the `Training` directory.

Example local path:

```python
data_dir = "/path/to/bratsglioma/Training"
```

Example Kaggle Notebook path:

```python
data_dir = "/kaggle/input/datasets/ahmedelzayat77/bratsglioma/Training"
```

---

### 2. MSD Brain Tumour Dataset

The Medical Segmentation Decathlon brain tumour dataset is also supported and can be downloaded from Kaggle:

[Download the MSD Brain Tumour Dataset from Kaggle](https://www.kaggle.com/datasets/trangnguyen25/msd-brain-tumour)

After extracting the dataset, the expected structure is:

```text
Task01_BrainTumour/
├── imagesTr/
├── labelsTr/
├── imagesTs/
└── dataset.json
```

Example path:

```python
data_dir = "/path/to/Task01_BrainTumour"
```

The MSD dataset may require a dataset loader or preprocessing configuration different from the BraTS Glioma dataset because its files are organized into separate image and label directories.



## Stage 1: Train the UNet Teacher Model

Before training the student model, first train the UNet teacher model using `train_teacher.py`.

Check and update the following settings in `train_teacher.py`:

Run the teacher training script:

```bash
python -u train_teacher.py
```

The best teacher checkpoint should be used for student knowledge distillation.

## Resume Teacher Training

When a previous teacher checkpoint is available, configure the checkpoint path in `train_teacher.py`.

For example:

```python
resume_path = "teacher_results/last_teacher.pth"
```

Then run the same training command:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train_teacher.py
```

The script should restore the model state, optimizer state, current epoch, and best validation score from the checkpoint.

## Stage 2: Train the Uni-Light Student Model

After teacher training is complete, configure the teacher checkpoint path in `train.py`.

For example:

```python
teacher_checkpoint = "teacher_results/best_teacher.pth"
```

The teacher model should be loaded and frozen during student training.

Run student knowledge distillation:

```bash
python -u train.py
```

During student training:

1. The trained UNet teacher model is loaded.
2. The teacher parameters are frozen.
3. The teacher generates predictions or intermediate representations.
4. The Uni-Light student is optimized using the segmentation loss and knowledge distillation losses.
5. The checkpoint with the best validation performance is saved.

## Resume Student Training

To resume interrupted student training, configure the student checkpoint path in `train.py`.

For example:

```python
resume_path = "student_results/last_student.pth"
```