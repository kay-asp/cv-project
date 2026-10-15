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

