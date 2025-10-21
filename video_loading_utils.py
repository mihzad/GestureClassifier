import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, Subset
from torchvision import tv_tensors
from torchvision.transforms import functional as F
import torchvision.transforms.v2 as v2

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
    def __init__(self, root_dir, per_img_transform: v2.Transform = None, production_ready=True):
        self.root_dir = root_dir

        if per_img_transform is not None:
            if not isinstance(per_img_transform, v2.Transform):
                raise TypeError("transform must be a torchvision.transforms.v2.Transform")
        self.transform = per_img_transform

        self.production_ready = production_ready

        # Map class names to indices
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # Build list of (video_dir_path, label_index)
        self.samples = []
        for cls_name in self.classes:
            class_dir = os.path.join(root_dir, cls_name)
            for vid_folder in sorted(os.listdir(class_dir)):
                vid_path = os.path.join(class_dir, vid_folder)
                if os.path.isdir(vid_path):
                    self.samples.append((vid_path, self.class_to_idx[cls_name]))

        self.targets = np.array([s[1] for s in self.samples])


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir, label = self.samples[idx]
        frame_paths = sorted([
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
        ])

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


# transform wrapper for separating val&test out of train augmentation
class TransformSubset(Subset):
    def __init__(self, dataset: VideoFramesFolderDataset, indices, per_img_transform: v2.Transform = None):
        super().__init__(dataset, indices)

        if not (per_img_transform is None) and not isinstance(per_img_transform, v2.Transform):
            raise TypeError("transform must be a torchvision.transforms.v2.Transform")
        self.transform = per_img_transform

        if hasattr(self.dataset, "targets"):
            self.targets = np.array(self.dataset.targets[indices])
        elif hasattr(self.dataset, "labels"):
            self.targets = np.array([self.dataset.labels[i] for i in self.indices])
        else:
            raise AttributeError(
                "Underlying dataset has no 'targets' or 'labels' attribute"
            )

    def __getitem__(self, idx):
        items = super().__getitem__(idx) #each item is (vid, label) where vid is TVTensors.Image(T,C,H,W)

        if self.transform is None:
            return items

        if isinstance(items, list):
            items = [((self.transform(vid)).permute(1,0,2,3), label) for vid, label in items]
        else: # tuple
            items = ((self.transform(items[0])).permute(1,0,2,3), items[1])

        return items

    def __getitems__(self, indices: list[int]):
        items = super().__getitems__(indices)

        return [((self.transform(vid)).permute(1, 0, 2, 3), label) for vid, label in items]


