def densenet121(num_classes, pretrained=True):

    weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None

    model = models.densenet121(weights=weights)

    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    return model
