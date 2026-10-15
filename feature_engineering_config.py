"""L3 VADER feature selection; separate from preprocessing config.py."""
from types import MappingProxyType

VADER_VERSION = "3.3.2"
VADER_COLUMNS = ("V_pos", "V_neg", "V_neu", "V_compound")
VADER_FEATURE_SETS = MappingProxyType({
    "V0": (),
    "V1": ("V_compound",),
    "V2": ("V_pos", "V_neg", "V_compound"),
})

def select_vader_columns(variant: str) -> tuple[str, ...]:
    """Use explicit approved feature names, never numeric-column inference."""
    if variant not in VADER_FEATURE_SETS:
        raise ValueError(f"Unknown VADER variant {variant!r}; expected V0, V1 or V2")
    return VADER_FEATURE_SETS[variant]

# L3-N: approved alternatives from label-blind N0/N0-B audits.
# The canonical original 13-word list remains in preprocessing config.py.
CONSERVATIVE_INTENSIFIER_WORDS = frozenset({
    "very", "really", "extremely", "absolutely", "totally",
    "completely", "highly", "incredibly",
})
N_COLUMNS = ("N_negation_count", "N_intensifier_original_count",
             "N_intensifier_conservative_count")
N_FEATURE_SETS = MappingProxyType({
    "N0": (),
    "N1": ("N_negation_count",),
    "N2": ("N_negation_count", "N_intensifier_original_count"),
    "N3": ("N_negation_count", "N_intensifier_conservative_count"),
})

def select_n_columns(variant: str) -> tuple[str, ...]:
    """Explicit label-free N-family feature whitelist."""
    if variant not in N_FEATURE_SETS:
        raise ValueError(f"Unknown N variant {variant!r}; expected N0, N1, N2 or N3")
    return N_FEATURE_SETS[variant]
