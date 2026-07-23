"""
Domain 25 — Information theory.

The project's gatekeeper. Entropy, divergence, and mutual-information estimators
feed the *signal gate*: a permutation test that decides whether each feature
carries statistically significant information about the target before any model
is trained. If a feature's mutual information is indistinguishable from noise,
it is dropped — however sophisticated the model that would consume it.
"""

from kalachakra.information.divergence import (
    jensen_shannon_divergence,
    kl_divergence,
    renyi_divergence,
)
from kalachakra.information.entropy import (
    conditional_entropy,
    cross_entropy,
    entropy,
    joint_entropy,
)
from kalachakra.information.feature_selection import mi_ranking, mrmr
from kalachakra.information.fisher import fisher_score
from kalachakra.information.mdl import aic, bic, gaussian_log_likelihood, two_part_mdl
from kalachakra.information.mutual_information import (
    conditional_mutual_information,
    information_gain,
    mutual_information,
    mutual_information_continuous,
    normalized_mutual_information,
)
from kalachakra.information.signal_vs_noise import (
    GateResult,
    mi_permutation_test,
    signal_gate,
)

__all__ = [
    # entropy
    "entropy",
    "joint_entropy",
    "conditional_entropy",
    "cross_entropy",
    # divergence
    "kl_divergence",
    "jensen_shannon_divergence",
    "renyi_divergence",
    # mutual information
    "mutual_information",
    "normalized_mutual_information",
    "conditional_mutual_information",
    "information_gain",
    "mutual_information_continuous",
    # feature selection
    "mi_ranking",
    "mrmr",
    "fisher_score",
    # mdl
    "aic",
    "bic",
    "two_part_mdl",
    "gaussian_log_likelihood",
    # the gate
    "GateResult",
    "mi_permutation_test",
    "signal_gate",
]
