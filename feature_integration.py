"""L4-M3 label-blind sparse block integration and distance calibration.

No K-means, label-based selection, or modification of L2/L3 outputs.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import sparse

TEXT_BLOCKS = frozenset(("word", "char"))
ENGINEERED_BLOCKS = frozenset(("V", "N", "S"))
ALLOWED_BLOCKS = TEXT_BLOCKS | ENGINEERED_BLOCKS

@dataclass(frozen=True)
class WeightingSpec:
    method: str = "W0"
    text_share: float = 0.70
    zero_distance: str = "reject"

    def __post_init__(self):
        if self.method not in ("W0", "W1", "W2"):
            raise ValueError("Unknown weighting method")
        if not 0 < self.text_share < 1:
            raise ValueError("text_share must lie strictly between 0 and 1")
        if self.zero_distance != "reject":
            raise ValueError("Only explicit rejection of zero-distance blocks is supported")

def _matrix(value, name):
    if name not in ALLOWED_BLOCKS:
        raise ValueError(f"Unknown block name: {name}")
    if sparse.issparse(value):
        x = value.tocsr().astype(np.float64)
        if not np.isfinite(x.data).all():
            raise ValueError(f"Nonfinite values in {name}")
    else:
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional matrix")
        if not np.isfinite(arr).all():
            raise ValueError(f"Nonfinite values in {name}")
        x = sparse.csr_matrix(arr)
    if x.shape[1] == 0:
        raise ValueError(f"Empty block {name}")
    return x

def _validate(blocks):
    if not isinstance(blocks, dict) or not blocks:
        raise ValueError("Provide at least one named block")
    if any("sentiment" in name.lower() or name.lower() == "row_id" for name in blocks):
        raise ValueError("Sentiment labels and row_id are not permitted blocks")
    converted = {name: _matrix(matrix, name) for name, matrix in blocks.items()}
    if len({matrix.shape[0] for matrix in converted.values()}) != 1:
        raise ValueError("All blocks must have identical row counts")
    if next(iter(converted.values())).shape[0] < 2:
        raise ValueError("At least two rows are required for calibration")
    return converted

def mean_squared_distances(blocks):
    """Exact mean squared distance for all ordered distinct row pairs.

    For n rows, E[||Xi-Xj||^2] = 2*n/(n-1) *
    (mean(||Xi||^2) - ||mean(X)||^2).
    Avoids pair sampling and remains efficient for sparse matrices.
    """
    matrices = _validate(blocks)
    distances = {}
    for name, x in matrices.items():
        n = x.shape[0]
        mean_norm_sq = float(x.multiply(x).sum()) / n
        column_mean = np.asarray(x.mean(axis=0)).ravel()
        mean_sq = float(np.dot(column_mean, column_mean))
        distances[name] = max(0.0, 2.0 * n / (n - 1) * (mean_norm_sq - mean_sq))
    return distances

def _targets(names, spec):
    if spec.method == "W1":
        return {name: 1.0 / len(names) for name in names}
    text = [name for name in names if name in TEXT_BLOCKS]
    engineered = [name for name in names if name in ENGINEERED_BLOCKS]
    if text and engineered:
        return {name: (spec.text_share / len(text) if name in text
                       else (1 - spec.text_share) / len(engineered))
                for name in names}
    # No group to receive the other group's share: redistribute over active blocks.
    return {name: 1.0 / len(names) for name in names}

class BlockIntegrator:
    """Fit block coefficients once; reuse for later aligned data."""
    def __init__(self, spec=None):
        self.spec = spec or WeightingSpec()
        self.names_ = None
        self.weights_ = None
        self.distances_ = None
        self.targets_ = None
        self.dimensions_ = None

    def fit(self, blocks):
        matrices = _validate(blocks)
        self.names_ = tuple(matrices)
        self.dimensions_ = {name: x.shape[1] for name, x in matrices.items()}
        self.distances_ = mean_squared_distances(matrices)
        if self.spec.method == "W0":
            self.targets_ = None
            self.weights_ = {name: 1.0 for name in matrices}
        else:
            self.targets_ = _targets(self.names_, self.spec)
            if any(d <= 1e-14 for d in self.distances_.values()):
                raise ValueError("Zero-distance block cannot receive a positive target")
            # Set target expected squared distances. Common scaling is arbitrary;
            # chosen so total expected squared distance equals 1.
            self.weights_ = {name: float(np.sqrt(self.targets_[name] / d))
                             for name, d in self.distances_.items()}
        return self

    def transform(self, blocks):
        if self.weights_ is None:
            raise RuntimeError("Fit the integrator before transform")
        matrices = _validate(blocks)
        if tuple(matrices) != self.names_:
            raise ValueError("Block names or ordering differ from fit")
        for name, x in matrices.items():
            if x.shape[1] != self.dimensions_[name]:
                raise ValueError(f"Feature dimension changed for {name}")
        return sparse.hstack(
            [matrices[name] * self.weights_[name] for name in self.names_],
            format="csr"
        )

    def fit_transform(self, blocks):
        return self.fit(blocks).transform(blocks)

    def manifest(self):
        if self.weights_ is None:
            raise RuntimeError("Integrator is not fitted")
        return {
            "method": self.spec.method,
            "text_share": self.spec.text_share if self.spec.method == "W2" else None,
            "block_order": list(self.names_),
            "dimensions": self.dimensions_,
            "calibration_mean_squared_distances": self.distances_,
            "target_shares": self.targets_,
            "weights": self.weights_,
            "calibration_method": "exact_all_distinct_pairs"
        }

def save_integrated(matrix, manifest, output_prefix, provenance):
    """Save sparse output and mandatory source/code provenance outside GitHub."""
    if not provenance or not provenance.get("git_revision") or not provenance.get("sources"):
        raise ValueError("Provide git_revision and source SHA256 mappings")
    for name, sha in provenance["sources"].items():
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha.lower()):
            raise ValueError(f"Invalid SHA256 for {name}")
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    matrix_path = Path(str(prefix) + ".npz")
    manifest_path = Path(str(prefix) + ".json")
    sparse.save_npz(matrix_path, matrix.tocsr())
    payload = {**manifest, "provenance": provenance,
               "shape": list(matrix.shape), "nnz": int(matrix.nnz),
               "matrix_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest()}
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(matrix_path), str(manifest_path)
