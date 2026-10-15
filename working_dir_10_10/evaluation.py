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

    correct = {species: 0 for species in species_list}
    total = {species: 0 for species in species_list}

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu()

            for pred, label in zip(preds, labels):
                species = species_list[label.item()]
                total[species] += 1
                if pred.item() == label.item():
                    correct[species] += 1

    return {
        species: (correct[species] / total[species] if total[species] > 0 else float("nan"))
        for species in species_list
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

    Scans the dataloader once to find the most frequent label, then reports
    the accuracy that predicting only that label would achieve — i.e. its
    empirical frequency.

    Returns:
        float: majority-class accuracy
    """
    label_counts = Counter()
    total = 0

    for _, labels in dataloader:
        for label in labels:
            label_counts[label.item()] += 1
            total += 1

    if total == 0:
        return 0.0

    _, majority_count = label_counts.most_common(1)[0]
    return majority_count / total


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
