import os
import matplotlib.pyplot as plt
from collections import Counter
from torchvision import datasets


def detect_splits(root):

    splits = {}
    for d in os.listdir(root):
        path = os.path.join(root, d)

        if not os.path.isdir(path):
            continue

        name = d.lower()

        if name in ["train", "training"]:
            splits["train"] = path

        elif name in ["test", "testing"]:
            splits["test"] = path

        elif name in ["val", "valid", "validation"]:
            splits["val"] = path

    return splits


def analyze_dataset(dataset_root):

    splits = detect_splits(dataset_root)

    if "train" not in splits:
        raise ValueError("Train folder not found!")

    train_ds = datasets.ImageFolder(splits["train"])

    class_counts = Counter(train_ds.targets)
    class_names = train_ds.classes

    print("\n================ DATASET REPORT ================\n")

    print(f"Train path: {splits['train']}")
    if "val" in splits:
        print(f"Val path: {splits['val']}")
    if "test" in splits:
        print(f"Test path: {splits['test']}")

    print(f"\nNumber of classes: {len(class_names)}")
    print(f"Total training images: {len(train_ds)}\n")

    print("Class distribution:")
    for i, count in class_counts.items():
        print(f"  {class_names[i]}: {count}")

    # imbalance check
    max_count = max(class_counts.values())
    min_count = min(class_counts.values())

    imbalance_ratio = max_count / min_count

    print(f"\nImbalance ratio: {imbalance_ratio:.2f}")

    if imbalance_ratio > 3:
        print("⚠ WARNING: Dataset is highly imbalanced!")

    return splits, train_ds


def show_samples(train_ds, num_classes=5):

    import numpy as np

    fig, axes = plt.subplots(1, num_classes, figsize=(15, 5))

    class_indices = {}

    for idx, label in enumerate(train_ds.targets):
        if label not in class_indices:
            class_indices[label] = idx

        if len(class_indices) >= num_classes:
            break

    for i, (label, idx) in enumerate(list(class_indices.items())[:num_classes]):

        img, _ = train_ds[idx]

        img = img.permute(1, 2, 0).numpy()

        axes[i].imshow(img)
        axes[i].set_title(train_ds.classes[label])
        axes[i].axis("off")

    plt.show()
