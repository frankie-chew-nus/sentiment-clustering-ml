"""L4-M2 label-blind engineered-feature scaling specifications."""
from dataclasses import dataclass

V1 = ("V_compound",)
V2 = ("V_pos", "V_neg", "V_compound")
N1 = ("N_negation_count",)
N2 = ("N_negation_count", "N_intensifier_original_count")
N3 = ("N_negation_count", "N_intensifier_conservative_count")
S1 = ("S_exclamation_count",)
S2 = ("S_exclamation_count", "S_uppercase_ratio")
S3 = ("S_exclamation_count", "S_uppercase_ratio", "S_emoji_count")

FEATURE_SETS = {
    "V0": (), "V1": V1, "V2": V2,
    "N0": (), "N1": N1, "N2": N2, "N3": N3,
    "S0": (), "S1": S1, "S2": S2, "S3": S3,
}
COUNT_COLUMNS = frozenset(N2 + N3 + S3) - {"S_uppercase_ratio"}
BOUNDED_COLUMNS = frozenset(("V_pos", "V_neg", "V_compound", "S_uppercase_ratio"))

@dataclass(frozen=True)
class ScalingSpec:
    method: str = "bounded"  # bounded, log_only, standard
    percentile: float = 95.0
    positive_fallback: bool = True
    clip: bool = True

    def __post_init__(self):
        if self.method not in ("bounded", "log_only", "standard"):
            raise ValueError("Unknown scaling method")
        if not 0 < self.percentile <= 100:
            raise ValueError("percentile must be in (0,100]")
