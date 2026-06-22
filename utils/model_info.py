
def print_trainable_layers(model):

    print("\nTrainable parameters:\n")

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
