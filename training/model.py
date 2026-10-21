import torch.nn as nn
from pytorchvideo.models.hub import x3d_s, x3d_m

def create_model(num_classes):
    model = x3d_m(pretrained=True)

    #replacing the classifier head
    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_features, num_classes)

    #correcting running_stats for tiny batches
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.momentum = 5e-4


    model.cuda(0)
    return model