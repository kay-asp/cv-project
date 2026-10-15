"""
Evaluation metrics: accuracy, depth-degradation, and baseline comparisons.

Responsibilities:
  - Per-species accuracy on shallow and deep test sets
  - Degradation score (accuracy_shallow − accuracy_deep)
  - Majority-class baseline predictor
  - Control-model comparison (trained on all depths)
"""

from collections import Counter

import torch

import config


def per_species_accuracy(model, dataloader, species_list):
    """Compute classification accuracy for each species.

    Args:
        model: Trained classifier.
        dataloader: Test DataLoader yielding (images, labels) batches, where
            labels are integer indices into species_list.
        species_list: List of species to evaluate (index order must match
            the label indices used when the model was trained).

    Returns:
        dict mapping species -> accuracy (float). A species with zero
        examples in this dataloader maps to float('nan').
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    n = len(species_list)
    correct = torch.zeros(n, dtype=torch.long)
    total = torch.zeros(n, dtype=torch.long)

    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            preds = model(images).argmax(dim=1).cpu()

            total += torch.bincount(labels, minlength=n)
            correct += torch.bincount(labels[preds == labels], minlength=n)

    return {
        species: (correct[i].item() / total[i].item() if total[i] > 0 else float("nan"))
        for i, species in enumerate(species_list)
    }


def compute_degradation(shallow_acc, deep_acc):
    """Degradation = accuracy_shallow − accuracy_deep, per species.

    Args:
        shallow_acc: dict species -> accuracy on shallow test set.
        deep_acc: dict species -> accuracy on deep test set.

    Returns:
        dict mapping species -> degradation score. Species missing from
        deep_acc map to float('nan') rather than being silently dropped.
    """
    return {
        species: shallow_acc[species] - deep_acc.get(species, float("nan"))
        for species in shallow_acc
    }


def majority_class_baseline(dataloader):
    """Accuracy of a majority-class predictor (trivial baseline).

    Counts the dataset's labels to find the most frequent one, then reports
    the accuracy that predicting only that label would achieve — i.e. its
    empirical frequency.

    Returns:
        float: majority-class accuracy
    """
    labels = dataloader.dataset.labels
    if not labels:
        return 0.0

    _, majority_count = Counter(labels).most_common(1)[0]
    return majority_count / len(labels)


def evaluate_control_model(control_model, shallow_loader, deep_loader, species_list):
    """Evaluate the all-depths control model to disentangle
    distribution shift from intrinsic deep-image difficulty.

    Returns:
        dict with keys 'shallow_acc', 'deep_acc', 'degradation' — each a
        dict keyed by species, matching the shape of the main model's
        results so they can be plotted side by side.
    """
    shallow_acc = per_species_accuracy(control_model, shallow_loader, species_list)
    deep_acc = per_species_accuracy(control_model, deep_loader, species_list)
    degradation = compute_degradation(shallow_acc, deep_acc)

    return {
        "shallow_acc": shallow_acc,
        "deep_acc": deep_acc,
        "degradation": degradation,
    }
