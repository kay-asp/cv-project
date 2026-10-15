#!/bin/bash
#SBATCH --job-name=cv-deepsea-modeling
#SBATCH --output=modeling_%j.log
#SBATCH --error=modeling_%j.err
#SBATCH --partition=biggpu              
#SBATCH --cpus-per-task=16           
#SBATCH --mem=64G
#SBATCH --time=24:00:00              

#Edit this for your setup
PROJECT_DIR="$HOME/cv-project"     

cd "$PROJECT_DIR"

# One-time environment setup (comment out after the first successful run):
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# source venv/bin/activate

# First run on this cluster: do the full crop download (slow, resumable).
# Once crops exist on disk, REMOVE --full-crops from later reruns —
# training will reuse them instead of re-downloading.
python run_modeling.py --project-dir "$PROJECT_DIR" --full-crops --crop-workers 8

# Example follow-up run once crops already exist and you just want to
# retrain or pick up from a checkpoint:
# python run_modeling.py --project-dir "$PROJECT_DIR"
