
def freeze_model(model):
    """
    Freeze all parameters.
    """
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_layers(model, layer_names):
    """
    Unfreeze only selected layers.
    """

    if "all" in layer_names:
        for param in model.parameters():
            param.requires_grad = True
        return

    for name, module in model.named_modules():
        if name in layer_names:
            for param in module.parameters():
                param.requires_grad = True
