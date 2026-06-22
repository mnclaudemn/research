import torch

def dino_vit():

    model = torch.hub.load(
        'facebookresearch/dino:main',
        'dino_vits16'
    )

    return model
