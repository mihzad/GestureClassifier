from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from torchvision import tv_tensors
from torchvision.transforms import functional as F
import torchvision.transforms.v2 as v2

from scripts.weighted_sampling_distributor import analyze_weaknesses_produce_weights

class VideoFramesFolderDataset(Dataset):
    '''
        per-frame video loader. the folder should have structure:
        folder/class/v1 (each video in its own subfolder)/[collection of frames]
        :parameter:
        :param root_dir: self-explanatory.
        :param per_img_transform: (Default: None) - V2(!) transform, that will be applied to each frame of
            the video. v2 is crucial for same transform per-frame.
        :param production_ready: (Default: True) - if false, returns video as collection of frame tensors [T,C,H,W] instead of
            stacking across temporal dim and creating training-ready [C,T,H,W] tensor. Useful for
            further subset-specific transforms inside external wrappers.
    '''
    def __init__(self, root_dir: Path, per_img_transform: v2.Transform = None, production_ready=True):
        self.root_dir = root_dir

        if per_img_transform is None:
            per_img_transform = v2.Compose([
                v2.Resize(size=(224,224)),
                v2.ToDtype(torch.float32, scale=True)
                ])
        if not isinstance(per_img_transform, v2.Transform):
            raise TypeError("transform must be a torchvision.transforms.v2.Transform")
        self.transform = per_img_transform

        self.production_ready = production_ready

        # Map class names to indices
        self.classes = sorted([d.name for d in root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Build list of (video_dir_path, label_index)
        self.samples = []
        for cls_name in self.classes:
            class_dir = root_dir / cls_name

            for vid_path in sorted(class_dir.iterdir()):
                if vid_path.is_dir():
                    self.samples.append((vid_path, self.class_to_idx[cls_name]))

        self.targets = np.array([s[1] for s in self.samples])


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir, label = self.samples[idx]
        frame_paths = sorted([f for f in video_dir.iterdir() if f.is_file()])


        frames = []

        for path in frame_paths:
            img = Image.open(path).convert('RGB')
            frames.append(F.to_tensor(img))

        tensorframes = tv_tensors.Image(torch.stack(frames))
        if self.transform is not None:
            tensorframes = self.transform(tensorframes)

        if self.production_ready:
            video = tensorframes.permute(1, 0, 2, 3)  # (T, C, H, W) => (C,T,H,W)
        else:
            video = tensorframes  # (T, C, H, W)

        return video, label


def create_dataloaders(training_set: VideoFramesFolderDataset,
                        validation_set: VideoFramesFolderDataset,
                        batch_size, num_workers,
                        additional_scaler_statistics_file: None | str):
    
        #======== creating balanced-batch dataloaders ========
    t_class_counts = np.bincount(training_set.targets)
    t_class_weights = 1.0 / t_class_counts #weights based on dataset imbalance

    # extra modifiers based on current situation and model`s weaknesses
    if additional_scaler_statistics_file is not None:
        t_class_modifiers = analyze_weaknesses_produce_weights(additional_scaler_statistics_file)
        t_class_weights *= t_class_modifiers

    t_sample_weights = [t_class_weights[t] for t in training_set.targets]
    train_sampler = WeightedRandomSampler(t_sample_weights, num_samples=len(t_sample_weights), replacement=True)
    train_loader = DataLoader(training_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)

    v_class_counts = np.bincount(validation_set.targets)
    v_class_weights = 1.0 / v_class_counts
    v_sample_weights = [v_class_weights[t] for t in validation_set.targets]
    val_sampler = WeightedRandomSampler(v_sample_weights, num_samples=len(v_sample_weights), replacement=True)
    val_loader = DataLoader(validation_set, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers)

    return train_loader, val_loader




