
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from sklearn.model_selection import KFold
from tqdm import tqdm
from utils.loss_metrics import BoundaryLoss


# ==========================================
# 1. 核心工具：智能划分与文件查找
# ==========================================

def get_5fold_split(data_dir, fold_idx, num_folds=5, seed=42):
    """
    自适应搜索路径并进行五折划分
    返回: (root_dir, train_list, val_list, mode)
    """
    # 搜索 imagesTr (MSD特征)
    search_path = os.path.join(data_dir, "**/imagesTr")
    found_dirs = glob.glob(search_path, recursive=True)

    case_list = []
    actual_root = data_dir
    mode = "Unknown"

    # --- 模式 A: MSD (扁平结构) ---
    if found_dirs:
        mode = "MSD"
        images_dir = found_dirs[0]
        actual_root = os.path.dirname(images_dir)  # MSD 根目录
        print(f"📂 [Detect] 模式: MSD (Found imagesTr at {images_dir})")

        if os.path.exists(images_dir):
            raw_files = os.listdir(images_dir)
            # MSD 列表存储文件名
            case_list = sorted([
                f for f in raw_files
                if (f.endswith('.nii.gz') or f.endswith('.nii')) and not f.startswith('.')
            ])

    # --- 模式 B: BraTS 2023 (嵌套结构) ---
    else:
        mode = "BraTS"
        actual_root = data_dir
        print(f"📂 [Detect] 模式: BraTS (Nested folders in {actual_root})")

        if os.path.exists(actual_root):
            raw_items = os.listdir(actual_root)
            # BraTS 列表存储完整路径
            case_list = sorted([
                os.path.join(actual_root, d)
                for d in raw_items
                if os.path.isdir(os.path.join(actual_root, d)) and not d.startswith('.')
            ])

    # --- 校验 ---
    if not case_list:
        raise FileNotFoundError(f"❌ 严重错误: 在 {data_dir} 下未找到有效数据 (BraTS/MSD 均未匹配)")

    print(f"📊 总病例数: {len(case_list)} | Mode: {mode}")

    # --- KFold ---
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    folds = list(kf.split(case_list))

    if fold_idx >= num_folds:
        raise ValueError(f"Fold index {fold_idx} out of range (max {num_folds - 1})")

    train_idx, val_idx = folds[fold_idx]
    train_cases = [case_list[i] for i in train_idx]
    val_cases = [case_list[i] for i in val_idx]

    # ⭐ 返回 mode 以便工厂函数使用
    return actual_root, train_cases, val_cases, mode


def find_nii_adaptive(case_dir, mode):
    """BraTS 专用：递归查找特定模态文件"""
    all_files = []
    for root, _, files in os.walk(case_dir):
        for f in files:
            if f.endswith(('.nii', '.nii.gz')):
                all_files.append(os.path.join(root, f))

    if not all_files: return None

    patterns = {
        'flair': ['-t2f', 'flair'],
        't1': ['-t1n', 't1.'],
        't1ce': ['-t1c', 't1ce', 't1gd'],
        't2': ['-t2w', 't2.'],
        'seg': ['-seg', 'seg']
    }

    candidates = []
    for f_path in all_files:
        f_name = os.path.basename(f_path).lower()
        if mode == 't1' and any(x in f_name for x in ['t1c', 't1ce', 't1gd']): continue
        if mode == 't2' and any(x in f_name for x in ['flair', 't2f']): continue

        if any(k in f_name for k in patterns[mode]):
            candidates.append(f_path)

    if not candidates: return None
    primary = {'flair': '-t2f', 't1': '-t1n', 't1ce': '-t1c', 't2': '-t2w', 'seg': '-seg'}
    for c in candidates:
        if primary[mode] in os.path.basename(c).lower(): return c
    return candidates[0]


def precompute_sdfs_unified(data_dir, case_list, out_dir, mode, roi_size=(128, 128, 128)):
    """
    统一的 SDF 预计算函数，支持 MSD 和 BraTS 两种模式
    """
    os.makedirs(out_dir, exist_ok=True)
    td, th, tw = roi_size
    print(f"🛠️ 正在预计算 SDFs (Mode: {mode}, Count: {len(case_list)})...")

    for case_item in tqdm(case_list, desc="SDF Precomputing"):
        # 1. 确定 Seg 路径和 Case Name
        if mode == "MSD":
            # MSD: case_item 是文件名
            case_name = case_item.replace('.nii.gz', '').replace('.nii', '')
            seg_path = os.path.join(data_dir, "labelsTr", case_item)
            if not os.path.exists(seg_path):
                seg_path = os.path.join(data_dir, "labels", case_item)  # 兼容
        else:
            # BraTS: case_item 是文件夹路径
            case_name = os.path.basename(case_item.rstrip('/'))
            seg_path = find_nii_adaptive(case_item, 'seg')

        out_path = os.path.join(out_dir, case_name + '.npz')

        # 2. 检查缓存
        if os.path.exists(out_path):
            try:
                with np.load(out_path) as d:
                    _ = d['sdfs']
                continue
            except:
                os.remove(out_path)  # 坏文件重做

        if not seg_path or not os.path.exists(seg_path):
            continue

        # 3. 读取 Seg 并统一处理
        try:
            seg = nib.load(seg_path).get_fdata().astype(np.int16)
        except Exception:
            continue

        # 标签映射逻辑
        if mode == "MSD":
            # MSD: 1:ED, 2:NCR, 3:ET -> WT(1+2+3), TC(2+3), ET(3)
            wt = ((seg == 1) | (seg == 2) | (seg == 3)).astype(np.uint8)
            tc = ((seg == 2) | (seg == 3)).astype(np.uint8)
            et = (seg == 3).astype(np.uint8)
        else:
            # BraTS 2023: 1:NCR, 2:ED, 3:ET -> WT(1+2+3), TC(1+3), ET(3)
            wt = ((seg == 1) | (seg == 2) | (seg == 3)).astype(np.uint8)
            tc = ((seg == 1) | (seg == 3)).astype(np.uint8)
            et = (seg == 3).astype(np.uint8)

        # 统一转置为 (3, D, H, W)
        # 注意：BraTS原始通常是 (H,W,D)，需要 transpose(2,0,1) 或根据实际情况
        # 这里统一转成 (Channel, D, H, W)
        seg_3ch = np.stack([wt, tc, et], axis=0)  # (3, H, W, D)
        seg_3ch = np.transpose(seg_3ch, (0, 3, 1, 2))

        # Pad & Crop
        _, d, h, w = seg_3ch.shape
        pd, ph, pw = max(0, td - d), max(0, th - h), max(0, tw - w)
        if pd or ph or pw:
            seg_3ch = np.pad(seg_3ch,
                             ((0, 0), (pd // 2, pd - pd // 2), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
        _, d, h, w = seg_3ch.shape
        sd, sh, sw = (d - td) // 2, (h - th) // 2, (w - tw) // 2
        seg_roi = seg_3ch[:, sd:sd + td, sh:sh + th, sw:sw + tw]

        # Compute SDF (float16)
        sdfs = np.zeros_like(seg_roi, dtype=np.float16)
        for ci in range(3):
            sdfs[ci] = BoundaryLoss.compute_sdf_np(seg_roi[ci]).astype(np.float16)

        np.savez_compressed(out_path, sdfs=sdfs)


# ==========================================
# 2. BraTS 2023 Dataset Class
# ==========================================

class BraTS2023Dataset(Dataset):
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
        c, d, h, w = data.shape
        td, th, tw = self.roi_size
        pd, ph, pw = max(0, td - d), max(0, th - h), max(0, tw - w)
        if pd or ph or pw:
            data = np.pad(data, ((0, 0), (pd // 2, pd - pd // 2), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
        _, d, h, w = data.shape
        sd, sh, sw = (d - td) // 2, (h - th) // 2, (w - tw) // 2
        return data[:, sd:sd + td, sh:sh + th, sw:sw + tw]

    def _augment_3d(self, image, seg, sdf=None):
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
        # Stack & Transpose: (4, H, W, D) -> (4, D, H, W)
        image = np.stack(modalities, axis=0)
        image = np.transpose(image, (0, 3, 1, 2))

        # Seg
        seg_file = find_nii_adaptive(case_dir, 'seg')
        seg = nib.load(seg_file).get_fdata().astype(np.int16)
        wt = ((seg == 1) | (seg == 2) | (seg == 3)).astype(np.float32)
        tc = ((seg == 1) | (seg == 3)).astype(np.float32)
        et = (seg == 3).astype(np.float32)
        # (3, H, W, D) -> (3, D, H, W)
        seg_3ch = np.stack([wt, tc, et], axis=0)
        seg_3ch = np.transpose(seg_3ch, (0, 3, 1, 2))

        image, seg_3ch = self._pad_crop(image), self._pad_crop(seg_3ch)

        sdf = np.zeros_like(seg_3ch)
        if self.sdf_dir:
            sdf_path = os.path.join(self.sdf_dir, f"{case_name}.npz")
            if os.path.exists(sdf_path):
                try:
                    sdf = np.load(sdf_path)['sdfs'].astype(np.float32)
                except:
                    pass

        if self.augment:
            image, seg_3ch, sdf = self._augment_3d(image, seg_3ch, sdf)

        ret = [torch.FloatTensor(image.copy()),torch.FloatTensor(seg_3ch.copy()),torch.FloatTensor(sdf.copy())]
        if self.return_case_name: ret.append(case_name)
        return tuple(ret)


# ==========================================
# 3. MSD Dataset Class
# ==========================================

class MSDBTSDataset(Dataset):
    def __init__(self, data_dir, case_names, sdf_dir=None, roi_size=(128, 128, 128), augment=False,
                 return_case_name=False):
        self.data_dir = data_dir
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
        if pd or ph or pw:
            data = np.pad(data, ((0, 0), (pd // 2, pd - pd // 2), (ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)))
        _, d, h, w = data.shape
        sd, sh, sw = (d - td) // 2, (h - th) // 2, (w - tw) // 2
        return data[:, sd:sd + td, sh:sh + th, sw:sw + tw]

    # ⚠️ 补充缺失的 Augmentation 方法
    def _augment_3d(self, image, seg, sdf=None):
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
        image += np.random.normal(0, 0.05, size=image.shape).astype(np.float32)
        return np.ascontiguousarray(image), np.ascontiguousarray(seg), sdf

    def __getitem__(self, idx):
        name = self.case_names[idx]
        img_path = os.path.join(self.data_dir, "imagesTr", name)

        try:
            img_data = nib.load(img_path).get_fdata().astype(np.float32)
        except Exception:
            raise FileNotFoundError(f"Read error: {img_path}")

        # MSD处理: (H,W,D,4) -> (4,D,H,W)
        if img_data.ndim == 4 and img_data.shape[-1] == 4:
            for c in range(4):
                img_data[..., c] = self._zscore_nonzero(img_data[..., c])
            # (H,W,D,C) -> (C,D,H,W)
            # 0,1,2,3 -> 3,2,0,1
            image = np.transpose(img_data, (3, 2, 0, 1))
        else:
            # 兼容可能的单通道或异常数据
            raise ValueError(f"Unexpected shape for MSD: {img_data.shape}")

        # Seg
        seg_path = os.path.join(self.data_dir, "labelsTr", name)
        if not os.path.exists(seg_path): seg_path = os.path.join(self.data_dir, "labels", name)
        seg = nib.load(seg_path).get_fdata().astype(np.int16)

        wt = ((seg == 1) | (seg == 2) | (seg == 3)).astype(np.float32)
        tc = ((seg == 2) | (seg == 3)).astype(np.float32)
        et = (seg == 3).astype(np.float32)
        seg_3ch = np.stack([wt, tc, et], axis=0)  # (3, H, W, D)
        seg_3ch = np.transpose(seg_3ch, (0, 3, 1, 2))  # (3, D, H, W)

        image, seg_3ch = self._pad_crop(image), self._pad_crop(seg_3ch)

        case_name = name.replace('.nii.gz', '').replace('.nii', '')
        sdf = np.zeros_like(seg_3ch)
        if self.sdf_dir:
            sdf_path = os.path.join(self.sdf_dir, f"{case_name}.npz")
            if os.path.exists(sdf_path):
                try:
                    sdf = np.load(sdf_path)['sdfs'].astype(np.float32)
                except:
                    pass

        if self.augment:
            image, seg_3ch, sdf = self._augment_3d(image, seg_3ch, sdf)

        ret = [torch.FloatTensor(image.copy()),torch.FloatTensor(seg_3ch.copy()),torch.FloatTensor(sdf.copy())]
        if self.return_case_name: ret.append(case_name)
        return tuple(ret)


# ==========================================
# 4. 工厂入口：自动选择 Dataset
# ==========================================

def get_datasets(data_dir, fold, num_folds=5, seed=42, sdf_dir=None, roi_size=(128, 128, 128)):
    """
    统一入口：
    1. 自动检测目录结构 (MSD vs BraTS)
    2. 自动预计算 SDF
    3. 自动实例化对应的 Dataset 类
    """
    # 1. 获取划分和模式
    root, train_list, val_list, mode = get_5fold_split(data_dir, fold, num_folds, seed)

    # 2. 预计算 SDF
    if sdf_dir:
        all_cases = train_list + val_list
        precompute_sdfs_unified(root, all_cases, sdf_dir, mode, roi_size)

    # 3. 实例化 Dataset
    if mode == "MSD":
        print("🏭 [Factory] 创建 MSDBTSDataset...")
        train_ds = MSDBTSDataset(root, train_list, sdf_dir=sdf_dir, roi_size=roi_size, augment=True, return_case_name=True)
        val_ds = MSDBTSDataset(root, val_list, sdf_dir=sdf_dir, roi_size=roi_size, augment=False, return_case_name=True)

    elif mode == "BraTS":
        print("🏭 [Factory] 创建 BraTS2023Dataset...")
        train_ds = BraTS2023Dataset(train_list, sdf_dir=sdf_dir, roi_size=roi_size, augment=True, return_case_name=True)
        val_ds = BraTS2023Dataset(val_list, sdf_dir=sdf_dir, roi_size=roi_size, augment=False, return_case_name=True)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return train_ds, val_ds