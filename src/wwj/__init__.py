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
    BayesLayerStats,
    bayes_analyze,
    bayes_analyze_matrix,
    bayes_summary,
    bayes_alpha_loss,
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
    "BayesLayerStats",
    "bayes_analyze",
    "bayes_analyze_matrix",
    "bayes_summary",
    "bayes_alpha_loss",
    "hierarchical_alpha",
    "hierarchical_analyze",
]
__version__ = "0.0.1"
