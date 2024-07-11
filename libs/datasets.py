import os
import cv2
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# train_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.CenterCrop((1024, 1024)),
#     transforms.RandomAffine(degrees=0, translate=(0.007, 0.007)),
#     transforms.RandomAffine(degrees=0, shear=(-15, 15)),
#     transforms.RandomRotation(degrees=(-25, 25)),
#     transforms.ColorJitter(saturation=0.5),
#     transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(254/255, 0, 0))
# ])

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop((1024, 1024)),
    transforms.RandomChoice([transforms.RandomHorizontalFlip(p=0.5),
                             transforms.RandomVerticalFlip(p=0.5)]),
    transforms.RandomAffine(degrees=0, translate=(0.007, 0.007)),
    transforms.RandomAffine(degrees=0, shear=(-15, 15)),
    transforms.RandomRotation(degrees=(-25, 25)),
    transforms.RandomChoice([transforms.ColorJitter(brightness=(0.5, 1.5)),
                             transforms.ColorJitter(contrast=0.5),
                             transforms.ColorJitter(saturation=0.5),
                             transforms.ColorJitter(hue=0.3)]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=(254/255, 0, 0))
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
])

class CancerDataset(Dataset):
    def __init__(self, df, transform=None):
        super(CancerDataset, self).__init__()
        self.df = df.copy()
        self.transform = transform
        self.path_to_png = "/home/FanHuang247817/train_images_png2/"
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.path_to_png, f"{self.df.loc[idx, 'patient_id']}/{self.df.loc[idx, 'image_id']}.png")
        img = cv2.imread(img_path)
        if self.transform:
            img = self.transform(img)
        label = self.df.loc[idx, "cancer"]
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        return img, label
    
    def __len__(self):
        return len(self.df)

class CustomBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, sampler, batch_size, drop_last, pos_indices) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.pos_indices = pos_indices
        
    def __iter__(self):
        batch = [random.choice(self.pos_indices)]
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                random.shuffle(batch)
                yield batch
                batch = [random.choice(self.pos_indices)]