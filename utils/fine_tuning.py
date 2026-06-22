# utils/fine_tuning.py

BLOCKS = {
    "resnet50": [ "layer1","layer2","layer3","layer4", "fc"],

    "densenet121": ["features.denseblock1", "features.denseblock2","features.denseblock3", "features.denseblock4", "classifier" ]
}


def unfreeze_last_n(model, model_name, n):
    """
    Unfreeze the last n blocks.

    Examples
    --------
    ResNet50:
        n=1 -> fc
        n=2 -> layer4 + fc
        n=3 -> layer3 + layer4 + fc

    DenseNet121:
        n=1 -> classifier
        n=2 -> denseblock4 + classifier
        n=3 -> denseblock3 + denseblock4 + classifier
    """

    blocks = BLOCKS[model_name]

    if n < 1 or n > len(blocks):
        raise ValueError(
            f"n must be between 1 and {len(blocks)}"
        )

    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # Select last n blocks
    blocks_to_train = blocks[-n:]

    # Unfreeze selected blocks
    for name, module in model.named_modules():
        if name in blocks_to_train:
            for p in module.parameters():
                p.requires_grad = True

    return blocks_to_train
