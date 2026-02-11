import glob
import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from sklearn.model_selection import KFold
from tqdm import tqdm

from utils.loss_metrics import BoundaryLoss


# ==========================================
# 1. Global Helper Functions for File Search
# ==========================================

def _rlist_case_files(case_dir):
    """Recursively list all .nii or .nii.gz files with size > 0."""
    out = []
    for root, _, files in os.walk(case_dir):
        for fn in files:
            if fn.endswith(".nii") or fn.endswith(".nii.gz"):
                p = os.path.join(root, fn)
                sz = os.path.getsize(p)
                if sz > 0:
                    # Store info: path, lowercase name, is_gz, size
                    out.append((p, fn.lower(), fn.endswith(".nii.gz"), sz))
    # Prioritize .nii.gz files
    out.sort(key=lambda x: (0 if x[2] else 1, x[0]))
    return out


def find_nii_adaptive(case_dir, mode):
    """
    Adaptive search for specific modality files in BraTS structure.
    Supports recursive searching and keyword matching.
    """
    all_files = _rlist_case_files(case_dir)
    if not all_files: return None

    patterns = {
        'flair': ['-t2f', 'flair'],
        't1': ['-t1n', 't1.'],
        't1ce': ['-t1c', 't1ce', 't1gd'],
        't2': ['-t2w', 't2.'],
        'seg': ['-seg', 'seg']
    }

    candidates = []
    for f_path, f_name, _, _ in all_files:
        # Exclusion logic: exclude t1ce when searching for t1, etc.
        if mode == 't1' and any(x in f_name for x in ['t1c', 't1ce', 't1gd']): continue
        if mode == 't2' and any(x in f_name for x in ['flair', 't2f']): continue

        if any(k in f_name for k in patterns[mode]):
            candidates.append(f_path)

    if not candidates: return None
    # Prioritize standard naming conventions
    primary = {'flair': '-t2f', 't1': '-t1n', 't1ce': '-t1c', 't2': '-t2w', 'seg': '-seg'}
    for c in candidates:
        if primary[mode] in os.path.basename(c).lower(): return c
    return candidates[0]


def precompute_sdfs_for_dataset(data_dir, case_list, out_dir, roi_size=(128, 128, 128)):
    os.makedirs(out_dir, exist_ok=True)
    print(f"正在为 MSD 数据集预计算 SDFs 到 {out_dir}...")

    td, th, tw = roi_size

    for name in tqdm(case_list, desc="SDF Precomputing"):
        if name.startswith("._") or name.startswith("."):
            continue
        # 1. 文件名处理：MSD 是文件名列表，不再是 BraTS 的文件夹路径
        base_name = name.replace('.nii.gz', '').replace('.nii', '')
        out_path = os.path.join(out_dir, base_name + '.npz')

        # 检查是否存在且完整
        if os.path.exists(out_path):
            try:
                with np.load(out_path) as data:
                    _ = data['sdfs']
                continue
            except:
                os.remove(out_path)

        # 2. 修改点：MSD 标签存储在固定的 labelsTr 目录下
        seg_path = os.path.join(data_dir, "labelsTr", name)
        if not os.path.exists(seg_path):
            continue

        seg_obj = nib.load(seg_path)
        seg = seg_obj.get_fdata().astype(np.int16)  # MSD 原始维度通常是 (H, W, D)

        # 3. 修改点：MSD 标签映射 (1:ED, 2:NCR, 3:ET)
        # 目标：WT=1+2+3, TC=2+3, ET=3
        wt = ((seg == 1) | (seg == 2) | (seg == 3)).astype(np.uint8)
        tc = ((seg == 2) | (seg == 3)).astype(np.uint8)
        et = (seg == 3).astype(np.uint8)

        seg_3ch = np.stack([wt, tc, et], axis=0)  # (3, H, W, D)

        # 4. 修改点：转置为 (C, D, H, W) 以符合 2023 框架逻辑
        seg_3ch = np.transpose(seg_3ch, (0, 3, 1, 2))

        # 5. 修改点：加入 Padding 保护，防止原始尺寸小于 128 导致裁剪失败
        c, d, h, w = seg_3ch.shape
        pd, ph, pw = max(0, td - d), max(0, th - h), max(0, tw - w)
        if pd or ph or pw:
            seg_3ch = np.pad(seg_3ch, (
                (0, 0),
                (pd // 2, pd - pd // 2),
                (ph // 2, ph - ph // 2),
                (pw // 2, pw - pw // 2)
            ), mode='constant')

        # 中心裁剪到 ROI_SIZE (完全复刻 2023 逻辑)
        _, curr_d, curr_h, curr_w = seg_3ch.shape
        sd = (curr_d - td) // 2
        sh = (curr_h - th) // 2
        sw = (curr_w - tw) // 2
        seg_roi = seg_3ch[:, sd:sd + td, sh:sh + th, sw:sw + tw]

        # 6. 修改点：使用 float16 存储，解决 Kaggle 磁盘空间不足问题
        sdfs = np.zeros_like(seg_roi, dtype=np.float16)
        for ci in range(3):
            # 依然调用你原代码中的 BoundaryLoss.compute_sdf_np
            sdf_val = BoundaryLoss.compute_sdf_np(seg_roi[ci])
            sdfs[ci] = sdf_val.astype(np.float16)

        np.savez_compressed(out_path, sdfs=sdfs)

    print("MSD SDF 预计算/检查完成。")


def get_5fold_split(data_dir, fold_idx, num_folds=5, seed=42):
    """
    自适应搜索路径并进行五折划分
    """
    # 1. 递归搜索 imagesTr 文件夹 (你的 glob 逻辑)
    search_path = os.path.join(data_dir, "**/imagesTr")
    found_dirs = glob.glob(search_path, recursive=True)

    if not found_dirs:
        # 保底方案
        images_dir = os.path.join(data_dir, "imagesTr")
    else:
        # 使用搜索到的第一个 imagesTr
        images_dir = found_dirs[0]

    # 获取数据集的真实根目录 (imagesTr 的父目录)
    # 这样可以确保 labelsTr 也能在这个目录下找到
    actual_root = os.path.dirname(images_dir)

    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"无法在 {data_dir} 下找到 imagesTr 文件夹。")

    # 2. 读取文件名并过滤隐藏文件 (你的过滤逻辑)
    all_files = sorted([
        f for f in os.listdir(images_dir)
        if (f.endswith('.nii.gz') or f.endswith('.nii')) and not f.startswith('.') and not f.startswith('._')
    ])

    if len(all_files) == 0:
        raise ValueError(f"在 {images_dir} 中没有找到有效的 .nii.gz 文件。")

    # 3. KFold 划分
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    folds = list(kf.split(all_files))
    train_idx, val_idx = folds[fold_idx]

    train_cases = [all_files[i] for i in train_idx]
    val_cases = [all_files[i] for i in val_idx]

    print(f"🚀 路径定位成功: {actual_root}")
    print(f"🚀 当前运行 Fold {fold_idx} | 训练集: {len(train_cases)} | 验证集: {len(val_cases)}")

    # 返回真实的根目录和划分的文件名列表
    return actual_root, train_cases, val_cases


# ==========================================
# 2. BraTS 2023 Dataset Class
# ==========================================

class BraTS2023Dataset(Dataset):
    """Dataset class for BraTS 2023 with recursive search and DWKD support."""

    def __init__(self, case_dirs, sdf_dir=None, roi_size=(128, 128, 128), augment=False, return_case_name=False):
        self.case_dirs = case_dirs
        self.sdf_dir = sdf_dir
        self.roi_size = roi_size
        self.augment = augment
        self.return_case_name = return_case_name

    def __len__(self):
        return len(self.case_dirs)

    def _zscore_nonzero(self, img):
        mask = img != 0
        if mask.any():
            v = img[mask]
            img[mask] = (v - v.mean()) / (v.std() + 1e-8)
        return img.astype(np.float32)

    def _pad_crop(self, data):
        """Pad or crop data to target ROI size."""
        c, d, h, w = data.shape
        td, th, tw = self.roi_size
        pd, ph, pw = max(0, td - d), max(0, th - h), max(0, tw - w)
        if pd > 0 or ph > 0 or pw > 0:
            data = np.pad(data, ((0, 0), (pd // 2, pd - pd // 2), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
        _, d, h, w = data.shape
        sd, sh, sw = (d - td) // 2, (h - th) // 2, (w - tw) // 2
        return data[:, sd:sd + td, sh:sh + th, sw:sw + tw]

    def _augment_3d(self, image, seg, sdf=None):
        """Synchronized 3D geometric and intensity augmentations."""
        if np.random.rand() < 0.5:
            image, seg = image[:, ::-1, :, :], seg[:, ::-1, :, :]
            if sdf is not None: sdf = sdf[:, ::-1, :, :]
        if np.random.rand() < 0.5:
            image, seg = image[:, :, ::-1, :], seg[:, :, ::-1, :]
            if sdf is not None: sdf = sdf[:, :, ::-1, :]
        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k=k, axes=(2, 3))
            seg = np.rot90(seg, k=k, axes=(2, 3))
            if sdf is not None: sdf = np.rot90(sdf, k=k, axes=(2, 3))
        # Intensity noise
        image += np.random.normal(0, 0.05, size=image.shape).astype(np.float32)
        return np.ascontiguousarray(image), np.ascontiguousarray(seg), sdf

    def __getitem__(self, idx):
        case_dir = self.case_dirs[idx]
        case_name = os.path.basename(case_dir.rstrip('/'))

        # Load 4 modalities
        modalities = []
        for mod in ['flair', 't1', 't1ce', 't2']:
            f = find_nii_adaptive(case_dir, mod)
            if not f: raise FileNotFoundError(f"Missing {mod} in {case_dir}")
            img = nib.load(f).get_fdata().astype(np.float32)
            modalities.append(self._zscore_nonzero(img))
        image = np.transpose(np.stack(modalities, axis=0), (0, 3, 1, 2))  # (C, D, H, W)

        # Load Segmentation and map labels: WT(1,2,3), TC(1,3), ET(3)
        seg_file = find_nii_adaptive(case_dir, 'seg')
        seg = nib.load(seg_file).get_fdata().astype(np.int16)
        wt = ((seg == 1) | (seg == 2) | (seg == 3)).astype(np.float32)
        tc = ((seg == 1) | (seg == 3)).astype(np.float32)
        et = (seg == 3).astype(np.float32)
        seg_3ch = np.transpose(np.stack([wt, tc, et], axis=0), (0, 3, 1, 2))

        image, seg_3ch = self._pad_crop(image), self._pad_crop(seg_3ch)

        # Load pre-computed SDF
        sdf = np.zeros_like(seg_3ch)
        if self.sdf_dir:
            sdf_path = os.path.join(self.sdf_dir, f"{case_name}.npz")
            if os.path.exists(sdf_path):
                sdf = np.load(sdf_path)['sdfs'].astype(np.float32)

        if self.augment:
            image, seg_3ch, sdf = self._augment_3d(image, seg_3ch, sdf)

        ret = [torch.FloatTensor(image), torch.FloatTensor(seg_3ch), torch.FloatTensor(sdf)]
        if self.return_case_name: ret.append(case_name)
        return tuple(ret)


# ==========================================
# 3. MSD Brain Tumor Dataset Class
# ==========================================

class MSDBTSDataset(Dataset):
    """Dataset class for MSD Task01 (Brain Tumor)."""

    def __init__(self, data_dir, case_names, sdf_dir=None, roi_size=(128, 128, 128), augment=False,
                 return_case_name=False):
        self.data_dir = data_dir  # Fixed: Added data_dir to handle paths properly
        self.case_names = case_names
        self.sdf_dir = sdf_dir
        self.roi_size = roi_size
        self.augment = augment
        self.return_case_name = return_case_name

    def __len__(self):
        return len(self.case_names)

    def _zscore_nonzero(self, img):
        mask = img != 0
        if mask.any():
            v = img[mask]
            img[mask] = (v - v.mean()) / (v.std() + 1e-8)
        return img.astype(np.float32)

    def _pad_crop(self, data):
        c, d, h, w = data.shape
        td, th, tw = self.roi_size
        pd, ph, pw = max(0, td - d), max(0, th - h), max(0, tw - w)
        if pd > 0 or ph > 0 or pw > 0:
            data = np.pad(data, ((0, 0), (pd // 2, pd - pd // 2), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
        _, d, h, w = data.shape
        sd, sh, sw = (d - td) // 2, (h - th) // 2, (w - tw) // 2
        return data[:, sd:sd + td, sh:sh + th, sw:sw + tw]

    def __getitem__(self, idx):
        name = self.case_names[idx]
        # Fixed: Use self.data_dir instead of global DATA_DIR
        img_path = os.path.join(self.data_dir, "imagesTr", name)
        img_data = nib.load(img_path).get_fdata().astype(np.float32)

        # MSD modality normalization and transpose to (C, D, H, W)
        for c in range(4):
            img_data[..., c] = self._zscore_nonzero(img_data[..., c])
        image = np.transpose(img_data, (3, 2, 0, 1))

        # MSD labels mapping: 1:ED, 2:NET, 3:ET
        seg_path = os.path.join(self.data_dir, "labelsTr", name)
        seg = nib.load(seg_path).get_fdata().astype(np.int16)
        wt = ((seg == 1) | (seg == 2) | (seg == 3)).astype(np.float32)
        tc = ((seg == 2) | (seg == 3)).astype(np.float32)
        et = (seg == 3).astype(np.float32)
        seg_3ch = np.transpose(np.stack([wt, tc, et], axis=0), (0, 3, 1, 2))

        image, seg_3ch = self._pad_crop(image), self._pad_crop(seg_3ch)

        # SDF loading
        case_name = name.replace('.nii.gz', '').replace('.nii', '')
        sdf = np.zeros_like(seg_3ch)
        if self.sdf_dir:
            sdf_path = os.path.join(self.sdf_dir, f"{case_name}.npz")
            if os.path.exists(sdf_path):
                sdf = np.load(sdf_path)['sdfs'].astype(np.float32)

        ret = [torch.FloatTensor(image), torch.FloatTensor(seg_3ch), torch.FloatTensor(sdf)]
        if self.return_case_name: ret.append(case_name)
        return tuple(ret)