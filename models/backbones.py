import torchvision.models as models
import torch.nn as nn
import timm
from models.dino import dino_vit

def get_model(name, num_classes):

    name = name.lower()

    if name == "resnet50":
        return resnet50(num_classes)

    if name == "densenet121":
        return densenet121(num_classes)

    if name == "vit":
        return vit_base(num_classes)

    if name == "dino":
        return dino_vit(num_classes)

    raise ValueError("Unknown model")
