"""
Evaluation metrics: accuracy, depth-degradation, and baseline comparisons.

Responsibilities:
  - Per-species accuracy on shallow and deep test sets
  - Degradation score (accuracy_shallow − accuracy_deep)
  - Majority-class baseline predictor
  - Control-model comparison (trained on all depths)
"""

import config


def per_species_accuracy(model, dataloader, species_list):
    """Compute classification accuracy for each species.

    Args:
        model: Trained classifier.
        dataloader: Test DataLoader.
        species_list: List of species to evaluate.

    Returns:
        dict mapping species -> accuracy (float)
    """
    raise NotImplementedError


def compute_degradation(shallow_acc, deep_acc):
    """Degradation = accuracy_shallow − accuracy_deep, per species.

    Args:
        shallow_acc: dict species -> accuracy on shallow test set.
        deep_acc: dict species -> accuracy on deep test set.

    Returns:
        dict mapping species -> degradation score
    """
    raise NotImplementedError


def majority_class_baseline(dataloader):
    """Accuracy of a majority-class predictor (trivial baseline).

    Returns:
        float: majority-class accuracy
    """
    raise NotImplementedError


def evaluate_control_model(control_model, shallow_loader, deep_loader, species_list):
    """Evaluate the all-depths control model to disentangle
    distribution shift from intrinsic deep-image difficulty.

    Returns:
        dict with control-model accuracies and degradation scores
    """
    raise NotImplementedError
