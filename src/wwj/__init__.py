# wwj public API.
from wwj.core import (
    LayerStats,
    analyze,
    analyze_matrix,
    alpha_loss,
    bootstrap_alpha_ci,
    fit_distributions,
    summary,
    _eigvals,
)
from wwj.bayes import (
    alpha_posterior,
    alpha_posterior_bma,
    model_posterior,
    ppc_pvalue,
)
from wwj.hierarchical import hierarchical_alpha, hierarchical_analyze

__all__ = [
    "LayerStats",
    "analyze",
    "analyze_matrix",
    "alpha_loss",
    "bootstrap_alpha_ci",
    "fit_distributions",
    "summary",
    "_eigvals",
    "alpha_posterior",
    "alpha_posterior_bma",
    "model_posterior",
    "ppc_pvalue",
    "hierarchical_alpha",
    "hierarchical_analyze",
]
__version__ = "0.0.1"
