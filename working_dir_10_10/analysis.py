"""
Statistical analysis and visualisation.

Responsibilities:
  - Spearman correlation between CRS and degradation
  - Scatter plot of CRS vs degradation per species
  - Baseline comparison plots (uniform, Gaussian, real Grad-CAM)
  - Summary tables / figures for reporting
"""


def spearman_correlation(crs_scores, degradation_scores):
    """Compute Spearman's ρ and p-value between CRS and degradation.

    Args:
        crs_scores: dict species -> CRS
        degradation_scores: dict species -> degradation

    Returns:
        rho (float), p_value (float)
    """
    raise NotImplementedError


def plot_crs_vs_degradation(crs_scores, degradation_scores, save_path=None):
    """Scatter plot: CRS (x) vs degradation (y) for each species.

    Args:
        crs_scores: dict species -> CRS
        degradation_scores: dict species -> degradation
        save_path: If provided, save the figure to this path.
    """
    raise NotImplementedError


def plot_baseline_comparison(real_crs, uniform_crs, gaussian_crs, save_path=None):
    """Compare real Grad-CAM CRS against uniform and Gaussian baselines.

    Shows that the real attribution is non-trivial.
    """
    raise NotImplementedError


def plot_control_comparison(main_degradation, control_degradation, save_path=None):
    """Compare degradation of shallow-only model vs all-depths control.

    Helps separate distribution shift from intrinsic difficulty.
    """
    raise NotImplementedError


def generate_summary_table(crs_scores, degradation_scores, rho, p_value):
    """Create a summary DataFrame with per-species CRS, degradation,
    and the overall Spearman result.

    Returns:
        pd.DataFrame
    """
    raise NotImplementedError
