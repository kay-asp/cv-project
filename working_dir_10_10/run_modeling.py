#!/usr/bin/env python3
"""
Deep-Sea Classification: Context Reliance & Depth Degradation
Script version of run_pipeline.ipynb, covering Setup & Data through Modeling.

Ported 1:1 from the notebook cells, with Colab-only cells (Drive mount,
!pip installs, interactive plt.show() sanity checks) removed or converted
to cluster-appropriate equivalents. Attribution & Analysis are still stubs
in attribution.py / analysis.py, so this script stops before them — add
those calls back in once that phase is implemented.

Usage:
    python run_modeling.py --project-dir /path/to/CV-project-directory
    python run_modeling.py --project-dir . --full-crops --crop-workers 16

First run on a new machine: pass --full-crops to download and cache all
crops (this is the ~173k-annotation download that timed out on Colab).
It's resumable — if the job dies partway, rerun the same command and
prepare_crops() only fetches what's still missing on disk.
"""

import argparse
import os
import sys
import time

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-dir", default=".",
                    help="Folder containing config.py, data.py, model.py, etc.")
    p.add_argument("--full-crops", action="store_true",
                    help="Run prepare_crops on the full dataset (slow, ~173k annotations). "
                         "Without this flag, only a 20-image smoke test runs, same as the "
                         "notebook default — use that first to confirm paths/credentials work.")
    p.add_argument("--crop-workers", type=int, default=8,
                    help="Parallel download threads for prepare_crops (default 8). "
                         "Raise for a faster cluster connection; lower if FathomNet starts rate-limiting.")
    p.add_argument("--epochs", type=int, default=None,
                    help="Override config.EPOCHS for both the classifier and control model.")
    p.add_argument("--skip-control", action="store_true",
                    help="Skip training the all-depths control model (classifier only).")
    return p.parse_args()


def main():
    args = parse_args()

    # ── 0 · Setup ────────────────────────────────────────────────────
    # (Colab's drive.mount() / !pip install cells are gone — on a cluster,
    # install requirements.txt into your environment before submitting the
    # job; see the accompanying submit_modeling.sh for the sbatch wrapper.)
    project_dir = os.path.abspath(args.project_dir)
    sys.path.insert(0, project_dir)
    os.chdir(project_dir)
    print(f"Project dir: {project_dir}")

    import config
    import data
    import model
    import evaluation
    # attribution, analysis intentionally not imported — still stubs

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("WARNING: no GPU visible to this job — check your SLURM --gres/--partition settings.")

    if args.epochs is not None:
        config.EPOCHS = args.epochs

    # ── 1 · Data loading & preprocessing ────────────────────────────
    print("\n--- Loading metadata ---")
    df, coco = data.load_fathomnet_metadata(config.TRAIN_PATH, config.DEPTH_PATH)
    print(f"Total images: {len(df)}")
    print(f"Depth range: {df['depth_m'].min():.0f}m - {df['depth_m'].max():.0f}m")

    # Split into shallow / deep, then restrict to species with enough
    # annotations on BOTH sides (a species only seen shallow or only seen
    # deep gives nothing to compare degradation against).
    threshold = config.DEPTH_SHALLOW_MAX
    shallow_df, deep_df = data.split_shallow_deep(df, threshold)
    shallow_df, deep_df = data.restrict_to_common_species(shallow_df, deep_df, min_n=5)
    all_df = pd.concat([shallow_df, deep_df], ignore_index=True)

    species_list = sorted(shallow_df["species"].unique())
    print(f"Shallow: {len(shallow_df)} | Deep: {len(deep_df)} | Species: {len(species_list)}")

    # Download + cache crops for every annotation we'll use (resumable).
    print("\n--- Preparing crops ---")
    if args.full_crops:
        data.prepare_crops(all_df, workers=args.crop_workers)
    else:
        print("Running smoke test only (20 images). Pass --full-crops for the real run.")
        data.prepare_crops(all_df, limit=20, workers=args.crop_workers)

    # Build DataLoaders (image-level train/val/test split, deterministic by seed)
    train_loader = data.get_dataloaders(shallow_df, split="train", species_list=species_list)
    val_loader = data.get_dataloaders(shallow_df, split="val", species_list=species_list)
    shallow_test_loader = data.get_dataloaders(shallow_df, split="test", species_list=species_list)
    deep_test_loader = data.get_dataloaders(deep_df, split="test", species_list=species_list)

    assert train_loader.species_list == species_list == deep_test_loader.species_list

    # ── 2 · Train classifier (shallow-depth only) ───────────────────
    print("\n--- Training classifier (shallow only) ---")
    classifier = model.build_classifier(num_classes=len(species_list))

    ckpt_path = config.MODEL_SAVE_PATH + "shallow_resnet50.pt"
    if os.path.exists(ckpt_path):
        print(f"Found existing checkpoint at {ckpt_path} - loading instead of retraining")
        classifier = model.load_checkpoint(classifier, ckpt_path)
    else:
        t0 = time.time()
        classifier, history = model.train(classifier, train_loader, val_loader)
        model.save_checkpoint(classifier, ckpt_path)
        print(f"Training took {(time.time() - t0) / 60:.1f} min, saved to {ckpt_path}")

    # ── 4 · Evaluation (the baseline/accuracy parts of Modeling) ────
    print("\n--- Evaluating classifier ---")
    shallow_acc = evaluation.per_species_accuracy(classifier, shallow_test_loader, species_list)
    deep_acc = evaluation.per_species_accuracy(classifier, deep_test_loader, species_list)
    degradation = evaluation.compute_degradation(shallow_acc, deep_acc)

    majority_acc = evaluation.majority_class_baseline(shallow_test_loader)
    print(f"Majority-class baseline: {majority_acc:.3f}")
    print(f"Mean shallow accuracy: {pd.Series(shallow_acc).mean():.3f}")
    print(f"Mean deep accuracy:    {pd.Series(deep_acc).mean():.3f}")

    # ── Control model (trained on all depths) ───────────────────────
    control_results = None
    if not args.skip_control:
        print("\n--- Training control model (all depths) ---")
        control_ckpt_path = config.MODEL_SAVE_PATH + "control_resnet50.pt"
        control = model.build_control_model(num_classes=len(species_list))

        all_loader = data.get_dataloaders(all_df, split="train", species_list=species_list)
        all_val_loader = data.get_dataloaders(all_df, split="val", species_list=species_list)

        if os.path.exists(control_ckpt_path):
            print(f"Found existing checkpoint at {control_ckpt_path} - loading instead of retraining")
            control = model.load_checkpoint(control, control_ckpt_path)
        else:
            t0 = time.time()
            control, _ = model.train(control, all_loader, all_val_loader)
            model.save_checkpoint(control, control_ckpt_path)
            print(f"Control training took {(time.time() - t0) / 60:.1f} min, saved to {control_ckpt_path}")

        control_results = evaluation.evaluate_control_model(
            control, shallow_test_loader, deep_test_loader, species_list
        )
        print(f"Control mean shallow accuracy: {pd.Series(control_results['shallow_acc']).mean():.3f}")
        print(f"Control mean deep accuracy:    {pd.Series(control_results['deep_acc']).mean():.3f}")

    print("\nModeling phase complete.")
    print("Not run (still stubs in attribution.py/analysis.py): Grad-CAM CRS, baseline "
          "attribution maps, Spearman correlation, plots, summary table.")

    # Results these hold, for whoever implements Attribution & Analysis next:
    #   classifier, species_list, shallow_test_loader, deep_test_loader,
    #   shallow_acc, deep_acc, degradation, majority_acc, control_results


if __name__ == "__main__":
    main()
