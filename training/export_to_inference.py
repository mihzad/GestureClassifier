from pathlib import Path

import torch
from torch import nn
from torch.export import Dim
from utils.checkpoint_management import load_model_only
from utils.common import curr_time, printshare
from model import create_model

from augmentations import create_transforms
from torchvision.transforms import v2
IMG_SIZE = 224
RAND_STATE = 44
N_CLASSES = 33
N_WORKSERS = 4

BEST_CHECKPOINT =  "private/best_checkpoint_90_test.pth"

class ResizeNormalizeWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.resize = v2.Resize(size=(IMG_SIZE, IMG_SIZE))

    def forward(self, video):
        video = self.resize(video)
        return self.model(video)

# class ResizeNormalizeWrapper(nn.Module):
#     def __init__(self, model, transform):
#         super().__init__()
#         self.model = model
#         self.transform = transform

#     def forward(self, video):
#         video = self.transform(video)
#         return self.model(video)

if __name__ == "__main__":

    model = create_model(N_CLASSES)

    model_w_weights = load_model_only(model, Path(BEST_CHECKPOINT))
    model_w_weights.eval().cuda()

    _, nontrain_transform = create_transforms(img_size=IMG_SIZE)

    # import numpy as np
    # nptensor = np.array(np.load("training/np_imagebatch_id111_label6.npy"))[np.newaxis, :]
    # tensor = torch.from_numpy(nptensor)
    # print(tensor)
    # tensor = tensor.cuda()
    # outputs = model_w_weights(tensor)
    # pred_vals, pred_classes = torch.max(outputs.data, 1)
    # print("TORCH MODEL")
    # print(pred_classes)
    # print(pred_vals)
    
    ready_wrapped_model = ResizeNormalizeWrapper(model=model_w_weights).cuda().eval()
    example_inputs = torch.rand(size=(2,3,16,256,360), dtype=torch.float32).cuda()


    dynamic_shapes_config = {
        "video": {
            0: Dim("batch", min=1, max=128),
            3: Dim("height", min=1),
            4: Dim("width", min=1),
        }
    }

    onnx_prog = torch.onnx.export(model=ready_wrapped_model,
                                   args=example_inputs,
                                     input_names=["video"],
                                       output_names=["output"],
                                        dynamic_shapes=dynamic_shapes_config,
                                        dynamo=True
                                        )

    onnx_prog.save('private/best_checkpoint.onnx')