# Deep-Sea Classification: Context Reliance & Depth Degradation

Does a deep-sea classification model's reliance on background context predict how badly it degrades at depth?

We test whether Grad-CAM attribution spillover outside an organism's segmentation mask (Context Reliance Score) correlates with misclassification as survey depth moves beyond the training distribution.

## Project Structure

```
├── config.py          # Hyperparameters, paths, depth thresholds
├── data.py            # FathomNet loading, cropping, depth splitting
├── model.py           # ResNet-50 setup and fine-tuning
├── attribution.py     # Grad-CAM and Context Reliance Score (CRS)
├── evaluation.py      # Accuracy, degradation, baselines
├── analysis.py        # Spearman correlation, scatter plots
├── run_pipeline.ipynb # Main Colab notebook (imports and runs the above)
```

## Collaboration in Google Colab

### Setup

1. Upload the entire project folder to a shared Google Drive folder.
2. Share that Drive folder with all team members (Editor access).
3. Open `run_pipeline.ipynb` in Google Colab from Drive.

### Avoiding conflicts

- **Do not edit the same `.py` file at the same time.** Colab saves to Drive, and the last save wins.
- Each person should own specific files:
  - Person A: `data.py`
  - Person B: `model.py`, `attribution.py`
  - Person C: `evaluation.py`, `analysis.py`
- `config.py` and `run_pipeline.ipynb` are shared — coordinate edits on these.
- Before editing a file, let your team know (e.g. a quick message in your group chat).

### Working with the .py files

The `.py` modules live on Drive alongside the notebook. The notebook mounts Drive and adds the project folder to the Python path so imports work directly. To edit a `.py` file, either:
- Open it in Colab (File → Open notebook → choose the .py file), or
- Edit it in the Drive web interface, or
- Download, edit locally, and re-upload.
