import torch
from torchinfo import summary
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from visual_gest_rec_own_architecture import MobileNet3D
from pytorchvideo.models.hub import x3d_s
import torch.nn as nn
n_classes = 33
def puppet_summarizer():
    # imagenet1k v2 weights for mobilenetv3-large from pytorch sourcecode
    model1 = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    model = MobileNet3D(n_classes=n_classes)

    net = x3d_s(pretrained=True)
    # replacing the classifier head
    in_features = net.blocks[-1].proj.in_features
    net.blocks[-1].proj = nn.Linear(in_features, n_classes)

    summary(
        net,
        input_size=(4, 3, 16, 224, 224),
        #input_size=(1, 3, 224, 224),
        depth=10,
        col_names=["input_size", "output_size", "num_params"],
        row_settings=["var_names", "depth"],
    )


def transfer_weights() -> dict:
    '''
    transfers weights from MobileNetV3-Large model (pretrained on ImageNet1k)
    to MobileNet3D model by copying weights across temporal dimension and normalizing them.
    the classifier WILL NOT BE COPIED.
    :return:
    MobileNet3D-compatible state dict. Use strict=False when loading it
    '''


    model_2d = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    model_3d_puppet = MobileNet3D(n_classes=n_classes)

    dict_2d = model_2d.state_dict()
    dict_3d = model_3d_puppet.state_dict()
    
    keys_2d = [k for k in dict_2d.keys() if not k.startswith("classifier.")]
    keys_3d = [k for k in dict_3d.keys() if not k.startswith("classifier.") and not k.startswith("mid_se.")]
    #mid_se will be filled in manually.

    if len(keys_2d) != len(keys_3d):
        raise ValueError(f"Key count mismatch: 2D has {len(keys_2d)}, 3D has {len(keys_3d)}."
                         f" Incompatible architectures?")

    to_3d_mapper = {k2d: k3d for k2d, k3d in zip(keys_2d, keys_3d)}

    for key2d in keys_2d:
        val2d = dict_2d[key2d]
        key3d = to_3d_mapper[key2d]
        if "fc" in key2d: #non-classifier and fc => SqueezeExcite ==> squeeze last two [1, 1] dims
            dict_3d[key3d] = val2d.squeeze(-1).squeeze(-1)
            
        elif len(val2d.shape) == 4 and ( key3d.startswith("blocks3D.") or key3d.startswith("conv1.") ):
            #not fc, in 3D pard of network and has 4-dim shape => 3Dconv ==> copy-normalize across T dim
            temporal_depth = dict_3d[key3d].shape[2]
            val_3d = (val2d.unsqueeze(2).repeat(1, 1, temporal_depth, 1, 1)) / temporal_depth
            dict_3d[key3d] = val_3d
        else: #the rest of shapes are just the same ==> copy-paste
            dict_3d[key3d] = val2d

    #mid_se
    mid_se_keys = ["mid_se.fc.0.weight", "mid_se.fc.0.bias", "mid_se.fc.2.weight", "mid_se.fc.2.bias"]
    dict_3d[mid_se_keys[0]] = torch.full_like(model_3d_puppet.mid_se.fc[0].weight, fill_value=0.5)
    dict_3d[mid_se_keys[1]] = torch.full_like(model_3d_puppet.mid_se.fc[0].bias, fill_value=0)
    dict_3d[mid_se_keys[2]] = torch.full_like(model_3d_puppet.mid_se.fc[2].weight, fill_value=0.5)
    dict_3d[mid_se_keys[3]] = torch.full_like(model_3d_puppet.mid_se.fc[2].bias, fill_value=0)


    return dict_3d

def convert_and_save():
    dict_3d = transfer_weights()

    torch.save(dict_3d, "../MobileNet3D_ImageNetBase.pth")

    print("\nSample keys from the generated 3D state_dict:")
    for k in dict_3d.keys():
        print(k, "                ", list(dict_3d[k].shape))


    model = MobileNet3D(n_classes=n_classes)
    missing, unexpected = model.load_state_dict(dict_3d, strict=False)
    print(f"missing: {missing};")
    print(f"unexpected: {unexpected}")
    print(len(dict_3d))


if __name__ == "__main__":
    puppet_summarizer()
    #convert_and_save()

