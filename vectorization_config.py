"""Approved L4-M1 experimental TF-IDF configurations (not clustering winners)."""
from dataclasses import dataclass

@dataclass(frozen=True)
class TfidfSpec:
    analyzer: str
    ngram_range: tuple[int, int]
    min_df: int = 2
    max_df: float = 0.95
    max_features: int = 30000
    sublinear_tf: bool = True
    norm: str = "l2"
    use_idf: bool = True
    lowercase: bool = False

WORD = TfidfSpec("word", (1, 2), max_features=30000)
CHAR = TfidfSpec("char", (3, 5), max_features=50000)
CHAR_WB = TfidfSpec("char_wb", (3, 5), max_features=50000)

# Explicit input selections. Combined alternatives isolate preprocessing effects.
REPRESENTATIONS = {
    "word_p1": (("word", "P1"),),
    "word_p2": (("word", "P2"),),
    "char_p0": (("char", "P0"),),
    "char_p1": (("char", "P1"),),
    "char_wb_p0": (("char_wb", "P0"),),
    "combined_p1_p1": (("word", "P1"), ("char", "P1")),
    "combined_p1_p0": (("word", "P1"), ("char", "P0")),
}
SPECS = {"word": WORD, "char": CHAR, "char_wb": CHAR_WB}
