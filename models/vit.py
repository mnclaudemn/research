def vit_base(num_classes, pretrained=True):

    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=pretrained,
        num_classes=num_classes
    )

    return model
