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

## Using Git with Google Colab

### First-time setup (run once per Colab session)

Open a new Colab notebook and run these cells:

```python
# Clone the repo
!git clone https://github.com/<your-org>/deepsea-crs.git
%cd deepsea-crs

# Configure git identity (use your own details)
!git config user.name "Your Name"
!git config user.email "you@example.com"
```

### Pulling the latest changes

At the start of every session, make sure you have the latest code:

```python
%cd /content/deepsea-crs
!git pull origin main
```

### Pushing your changes

After making edits:

```python
!git add -A
!git commit -m "describe what you changed"
!git push origin main
```


### Authentication

Colab won't have your GitHub credentials by default. The simplest option is a Personal Access Token (PAT):

1. Go to GitHub → Settings → Developer settings → Personal access tokens → Generate new token
2. When Colab prompts for a password during `git push`, paste the token instead

Alternatively, clone with the token in the URL (keep this private):

```python
!git clone https://<TOKEN>@github.com/<your-org>/deepsea-crs.git
```
