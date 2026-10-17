"""
Evaluation metrics: accuracy, depth-degradation, and baseline comparisons.

Responsibilities:
  - Per-species accuracy on shallow and deep test sets
  - Degradation score (accuracy_shallow − accuracy_deep)
  - Majority-class baseline predictor
  - Control-model comparison (trained on all depths)
"""

import os
from collections import Counter

import pandas as pd
import torch

import config


def predict(model, dataloader):
    """Run the model over a (non-shuffled) DataLoader and keep every prediction.

    Args:
        model: Trained classifier.
        dataloader: DataLoader built by data.get_dataloaders (has
            .species_list, and its dataset has image_ids / ann_ids).

    Returns:
        DataFrame with one row per crop: image_id, ann_id, species (true),
        pred (predicted species), confidence (softmax prob of pred), correct.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    labels, preds, confs = [], [], []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for images, batch_labels, *_ in dataloader:  # *_ skips masks if the loader has them
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            probs = model(images).float().softmax(dim=1)
            conf, pred = probs.max(dim=1)
            labels.append(batch_labels)
            preds.append(pred.cpu())
            confs.append(conf.cpu())

    species_list = dataloader.species_list
    labels, preds = torch.cat(labels).tolist(), torch.cat(preds).tolist()
    ds = dataloader.dataset
    return pd.DataFrame({
        "image_id": ds.image_ids,
        "ann_id": ds.ann_ids,
        "species": [species_list[i] for i in labels],
        "pred": [species_list[i] for i in preds],
        "confidence": torch.cat(confs).tolist(),
        "correct": [l == p for l, p in zip(labels, preds)],
    })


def accuracy_from_predictions(preds, species_list):
    """Per-species accuracy from a predict() DataFrame.

    Returns:
        dict mapping species -> accuracy (float). A species with zero
        examples maps to float('nan').
    """
    acc = preds.groupby("species")["correct"].mean()
    return {species: float(acc.get(species, float("nan"))) for species in species_list}


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
    return accuracy_from_predictions(predict(model, dataloader), species_list)


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
    shallow_preds = predict(control_model, shallow_loader)
    deep_preds = predict(control_model, deep_loader)
    shallow_acc = accuracy_from_predictions(shallow_preds, species_list)
    deep_acc = accuracy_from_predictions(deep_preds, species_list)
    degradation = compute_degradation(shallow_acc, deep_acc)

    return {
        "shallow_acc": shallow_acc,
        "deep_acc": deep_acc,
        "degradation": degradation,
        "shallow_preds": shallow_preds,
        "deep_preds": deep_preds,
    }


def save_results(species_list, shallow_preds, deep_preds, control_results=None,
                 results_dir=config.RESULTS_DIR):
    """Write per-species accuracy and per-image predictions to CSV.

    Files written to results_dir:
      per_species_accuracy.csv: one row per species with test counts,
          shallow/deep accuracy and degradation (plus control_* columns).
      predictions.csv: one row per test crop per model, with columns
          model ('shallow' | 'control'), split ('shallow' | 'deep'),
          image_id, ann_id, species, pred, confidence, correct.

    Args:
        species_list: Class order used by the models.
        shallow_preds, deep_preds: predict() output of the shallow-only model.
        control_results: evaluate_control_model() output, or None.
        results_dir: Output directory (created if missing).

    Returns:
        (per_species DataFrame, predictions DataFrame)
    """
    os.makedirs(results_dir, exist_ok=True)

    shallow_acc = accuracy_from_predictions(shallow_preds, species_list)
    deep_acc = accuracy_from_predictions(deep_preds, species_list)
    per_species = pd.DataFrame({
        "n_shallow_test": shallow_preds["species"].value_counts(),
        "n_deep_test": deep_preds["species"].value_counts(),
        "shallow_acc": shallow_acc,
        "deep_acc": deep_acc,
        "degradation": compute_degradation(shallow_acc, deep_acc),
    }).reindex(species_list)
    per_species[["n_shallow_test", "n_deep_test"]] = (
        per_species[["n_shallow_test", "n_deep_test"]].fillna(0).astype(int))

    runs = [("shallow", "shallow", shallow_preds), ("shallow", "deep", deep_preds)]
    if control_results is not None:
        for key in ("shallow_acc", "deep_acc", "degradation"):
            per_species["control_" + key] = pd.Series(control_results[key])
        runs += [("control", "shallow", control_results["shallow_preds"]),
                 ("control", "deep", control_results["deep_preds"])]
    per_species.index.name = "species"

    predictions = pd.concat(
        [p.assign(model=m, split=s) for m, s, p in runs], ignore_index=True)
    predictions = predictions[["model", "split", "image_id", "ann_id",
                               "species", "pred", "confidence", "correct"]]

    per_species_path = os.path.join(results_dir, "per_species_accuracy.csv")
    predictions_path = os.path.join(results_dir, "predictions.csv")
    per_species.to_csv(per_species_path)
    predictions.to_csv(predictions_path, index=False)
    print(f"Saved {len(per_species)} species to {per_species_path}")
    print(f"Saved {len(predictions)} predictions to {predictions_path}")
    return per_species, predictions
