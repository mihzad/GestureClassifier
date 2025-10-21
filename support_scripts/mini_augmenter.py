import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.transforms.v2 as v2
import os

from video_loading_utils import TransformSubset, VideoFramesFolderDataset

# Image size
img_size = 224

default_transform = v2.Compose([
    v2.RandomResizedCrop(
        size=(img_size, img_size),
        scale=(0.75, 1.0),
        ratio=(9.0 / 10.0, 10.0 / 9.0)
    ),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
])


def generate_valset(base_valset: TransformSubset | VideoFramesFolderDataset,
                    save_to, transform=default_transform, times_more=3):

    augmented_data = []
    augmented_targets = []

    for idx in range(len(base_valset)):
        vid, label = base_valset[idx]
        for _ in range(times_more):
            augmented_vid = transform(vid)
            augmented_data.append(augmented_vid)
            augmented_targets.append(label)

    # Stack into tensors
    augmented_data = torch.stack(augmented_data)
    augmented_targets = torch.tensor(augmented_targets)

    print(f"Final augmented val set shape: {augmented_data.shape}, labels: {augmented_targets.shape}")

    torch.save((augmented_data, augmented_targets), save_to)
    print("done.")

