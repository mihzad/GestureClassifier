from pathlib import Path

import torch

from utils.checkpoint_management import load_model_only
from utils.common import curr_time, printshare
from model import create_model



IMG_SIZE = 224
RAND_STATE = 44
N_CLASSES = 33
N_WORKSERS = 4

BEST_CHECKPOINT =  "private/best_checkpoint_90_test.pth"


model = create_model(N_CLASSES)

model_w_weights = load_model_only(model, BEST_CHECKPOINT)
model.eval()

example_inputs = (torch.randn(1,3,16,IMG_SIZE,IMG_SIZE,).cuda(),)
onnx_prog = torch.onnx.export(model=model_w_weights, args=example_inputs, dynamo=True)

onnx_prog.save('private/best_checkpoint.onnx')