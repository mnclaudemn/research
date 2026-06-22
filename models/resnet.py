def resnet50(num_classes, pretrained=True):

    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None

    model = models.resnet50(weights=weights)

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
