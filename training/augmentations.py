import torch
from torchvision.transforms import v2
import math

def create_transforms(img_size = 224):
    train_transform = v2.Compose([
        v2.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.75, 1.0),
            ratio=(7.0 / 8.0, 8.0 / 7.0)
        ),
        v2.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),

        # edge padding before + centercrop after rotation => corners filled.
        v2.Pad(padding_mode="edge", padding=math.ceil(img_size * 0.2)),
        v2.RandomRotation(
            degrees=15,
            interpolation=v2.InterpolationMode.BILINEAR
        ),
        v2.CenterCrop(img_size),

        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.GaussianNoise(mean=0, sigma=0.08),
        v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    ])

    nontrain_transform = v2.Compose([
        v2.Resize(size=(img_size, img_size)),
        v2.ToDtype(torch.float32, scale=True),
    ])
    return train_transform, nontrain_transform