import os
import cv2
import numpy as np
import torch
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

class MVTecDefectDataset(Dataset):
    def __init__(self, root_dir, object_class, split="train", transform=None):
        """
        Args:
            root_dir (str): Directory with MVTec AD dataset
            object_class (str): Object class (e.g., 'bottle', 'cable', etc.)
            split (str): 'train' or 'test'
            transform: Optional transform to be applied
        """
        # 确保路径使用正确的格式
        self.root_dir = os.path.normpath(root_dir)
        self.object_class = object_class
        self.split = split
        self.transform = transform
        
        print(f"初始化数据集: root_dir={self.root_dir}, object_class={object_class}, split={split}")
        
        # 获取缺陷类型
        self.defect_types = []
        if split == "train":
            defect_path = os.path.join(self.root_dir, object_class, "train", "defective")
            print(f"查找缺陷类型: {defect_path}")
            if os.path.exists(defect_path):
                self.defect_types = [d for d in os.listdir(defect_path) if os.path.isdir(os.path.join(defect_path, d))]
                print(f"找到缺陷类型: {self.defect_types}")
            else:
                print(f"警告: 目录不存在 {defect_path}")
                # 检查上级目录是否存在
                parent_dir = os.path.dirname(defect_path)
                if os.path.exists(parent_dir):
                    print(f"父目录 {parent_dir} 存在，包含: {os.listdir(parent_dir)}")
                else:
                    print(f"父目录 {parent_dir} 不存在")
                
                # 检查根目录
                if os.path.exists(self.root_dir):
                    root_contents = os.listdir(self.root_dir)
                    print(f"根目录 {self.root_dir} 存在，包含: {root_contents[:5]}... (共 {len(root_contents)} 项)")
                else:
                    print(f"根目录 {self.root_dir} 不存在")
        
        # 加载图像和掩码路径
        self.images = []
        self.masks = []
        
        # 处理测试集 (只加载good目录)
        if split == "test":
            good_dir = os.path.join(self.root_dir, object_class, "test", "good")
            print(f"查找测试集: {good_dir}")
            if os.path.exists(good_dir):
                good_files = sorted([f for f in os.listdir(good_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
                print(f"测试集中找到 {len(good_files)} 个good样本")
                
                for good_file in good_files:
                    good_path = os.path.join(good_dir, good_file)
                    self.images.append(good_path)
                    # 对于good图像，在训练/测试时生成随机掩码
                    self.masks.append(None)
            else:
                print(f"警告: 测试目录不存在 {good_dir}")
                
        # 加载缺陷图像 (训练集)
        else:
            for defect_type in self.defect_types:
                img_dir = os.path.join(self.root_dir, object_class, "train", "defective", defect_type)
                mask_dir = os.path.join(self.root_dir, object_class, "train", "defective_masks", defect_type)
                
                print(f"处理缺陷类型: {defect_type}")
                print(f"  图像目录: {img_dir}")
                print(f"  掩码目录: {mask_dir}")
                
                if os.path.exists(img_dir) and os.path.exists(mask_dir):
                    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
                    print(f"  找到 {len(img_files)} 个图像文件")
                    
                    matched = 0
                    for img_file in img_files:
                        img_path = os.path.join(img_dir, img_file)
                        
                        # 获取基本文件名(不含扩展名)
                        base_name = os.path.splitext(img_file)[0]
                        mask_file = f"{base_name}_mask.png"
                        mask_path = os.path.join(mask_dir, mask_file)
                        
                        # 如果找不到掩码，尝试其他模式
                        if not os.path.exists(mask_path):
                            possible_masks = [f for f in os.listdir(mask_dir) if base_name in f]
                            if possible_masks:
                                mask_path = os.path.join(mask_dir, possible_masks[0])
                                print(f"  使用替代掩码: {mask_path}")
                            else:
                                print(f"  警告: 找不到 {img_file} 的掩码，跳过")
                                continue
                        
                        self.images.append(img_path)
                        self.masks.append(mask_path)
                        matched += 1
                    
                    print(f"  成功匹配 {matched} 个图像-掩码对")
        
        print(f"总共加载 {len(self.images)} 个 {split} 图像")
        if len(self.images) == 0:
            print("警告: 没有加载到图像! 数据集为空。")
 
    def __len__(self):
        return len(self.images)
    
    def generate_random_mask(self, image_size):
        """Generate random rectangular masks for object loss"""
        mask = np.zeros(image_size, dtype=np.float32)
        num_rectangles = 30
        
        h, w = image_size
        for _ in range(num_rectangles):
            # Rectangle size between 3% and 25% of image dimensions
            min_size = int(min(h, w) * 0.03)
            max_size = int(min(h, w) * 0.25)
            
            # Random rectangle dimensions
            rect_h = np.random.randint(min_size, max_size)
            rect_w = np.random.randint(min_size, max_size)
            
            # Random position
            y = np.random.randint(0, h - rect_h)
            x = np.random.randint(0, w - rect_w)
            
            # Add rectangle to mask
            mask[y:y+rect_h, x:x+rect_w] = 1.0
            
        return mask
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load or generate mask
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = mask.astype(np.float32) / 255.0  # Normalize to [0, 1]
        else:
            # Generate random mask for good images
            mask = self.generate_random_mask((image.shape[0], image.shape[1]))
            
        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Create background by removing defect regions
        background = image * (1 - mask)
        
        # For object loss, create adjusted mask
        adjusted_mask = mask + 0.3 * (1 - mask) if mask_path is None else mask
        
        return {
            'image': image,
            'mask': mask,
            'background': background,
            'adjusted_mask': adjusted_mask,
            'is_defect': mask_path is not None,
            'object_class': self.object_class
        }

def get_data_loaders(root_dir, object_class, batch_size=4):
    # Training transformations
    train_transform = A.Compose([
        A.RandomScale(scale_limit=(0.0, 0.125), p=1.0),  # Random scale between 1.0 and 1.125
        A.Resize(512, 512),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ], additional_targets={'mask': 'mask', 'background': 'image', 'adjusted_mask': 'mask'})
    
    # Test transformations
    test_transform = A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ], additional_targets={'mask': 'mask', 'background': 'image', 'adjusted_mask': 'mask'})
    
    # Create datasets
    train_dataset = MVTecDefectDataset(
        root_dir=root_dir,
        object_class=object_class,
        split="train",
        transform=train_transform
    )
    
    test_dataset = MVTecDefectDataset(
        root_dir=root_dir,
        object_class=object_class,
        split="test",
        transform=test_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader