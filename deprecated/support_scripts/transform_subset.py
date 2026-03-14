import numpy as np

from torch.utils.data import Subset
from utils_data_loading import VideoFramesFolderDataset

import torchvision.transforms.v2 as v2

# unused: transform wrapper for separating val&test out of train augmentation
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