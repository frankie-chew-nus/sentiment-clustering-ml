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
