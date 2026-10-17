"""L4-M3 exact label-blind weighting and sparse integration tests."""
import json
import numpy as np
import pytest
from scipy import sparse
from feature_integration import (
    BlockIntegrator, WeightingSpec, mean_squared_distances, save_integrated
)

@pytest.fixture
def blocks():
    return {
        "word": sparse.csr_matrix([[1., 0], [0, 1], [1, 1], [0, 2]]),
        "char": sparse.csr_matrix([[0., 1], [2, 0], [1, 0], [1, 1]]),
        "V": np.array([[-.7], [.2], [.8], [0.]]),
        "N": np.array([[0], [1], [2], [1]]),
        "S": np.array([[0, .2], [1, 0], [0, .3], [2, .1]]),
    }

def test_distance_identity(blocks):
    distances = mean_squared_distances(blocks)
    for name, x in blocks.items():
        a = x.toarray() if sparse.issparse(x) else x
        brute = np.mean([
            np.sum((a[i] - a[j]) ** 2)
            for i in range(len(a)) for j in range(len(a)) if i != j
        ])
        assert np.isclose(distances[name], brute, rtol=1e-12)

@pytest.mark.parametrize("method", ["W0", "W1", "W2"])
def test_sparse_and_manifest(blocks, method):
    integrator = BlockIntegrator(WeightingSpec(method=method))
    x = integrator.fit_transform(blocks)
    assert sparse.isspmatrix_csr(x)
    assert x.shape == (4, 8)
    assert np.isfinite(x.data).all()
    assert integrator.manifest()["method"] == method

def test_w0_identity(blocks):
    integ = BlockIntegrator(WeightingSpec("W0")).fit(blocks)
    assert set(integ.weights_.values()) == {1.0}

@pytest.mark.parametrize("method,expected", [
    ("W1", {"word": .2, "char": .2, "V": .2, "N": .2, "S": .2}),
    ("W2", {"word": .35, "char": .35, "V": .1, "N": .1, "S": .1}),
])
def test_calibrated_shares(blocks, method, expected):
    integ = BlockIntegrator(WeightingSpec(method)).fit(blocks)
    for name, target in expected.items():
        d = integ.distances_[name] * integ.weights_[name] ** 2
        assert np.isclose(d, target, atol=1e-12)

def test_missing_family_redistribution(blocks):
    subset = {"word": blocks["word"], "V": blocks["V"]}
    integ = BlockIntegrator(WeightingSpec("W2")).fit(subset)
    assert integ.targets_["word"] == pytest.approx(.7)
    assert integ.targets_["V"] == pytest.approx(.3)
    only_text = {"word": blocks["word"], "char": blocks["char"]}
    assert BlockIntegrator(WeightingSpec("W2")).fit(only_text).targets_ == {
        "word": .5, "char": .5}

def test_zero_distance_rejected(blocks):
    blocks["N"] = np.ones((4, 1))
    with pytest.raises(ValueError, match="Zero-distance"):
        BlockIntegrator(WeightingSpec("W1")).fit(blocks)

def test_row_mismatch(blocks):
    blocks["V"] = blocks["V"][:-1]
    with pytest.raises(ValueError, match="row counts"):
        BlockIntegrator().fit(blocks)

def test_nonfinite(blocks):
    blocks["V"][0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        BlockIntegrator().fit(blocks)

def test_unknown_and_leakage(blocks):
    blocks["airline_sentiment"] = np.ones((4, 1))
    with pytest.raises(ValueError, match="Sentiment"):
        BlockIntegrator().fit(blocks)

def test_transform_requires_fit(blocks):
    with pytest.raises(RuntimeError, match="Fit"):
        BlockIntegrator().transform(blocks)

def test_dimension_change(blocks):
    integ = BlockIntegrator().fit(blocks)
    blocks["word"] = sparse.csr_matrix(np.ones((4, 3)))
    with pytest.raises(ValueError, match="dimension"):
        integ.transform(blocks)

def test_save_requires_provenance(blocks, tmp_path):
    x = BlockIntegrator().fit_transform(blocks)
    with pytest.raises(ValueError, match="git_revision"):
        save_integrated(x, {}, tmp_path / "x", {})

def test_save_roundtrip(blocks, tmp_path):
    integ = BlockIntegrator().fit(blocks)
    x = integ.transform(blocks)
    source = {"master": "a" * 64}
    a, b = save_integrated(x, integ.manifest(), tmp_path / "x",
                           {"git_revision": "test-revision", "sources": source})
    assert sparse.load_npz(a).shape == x.shape
    assert json.load(open(b))["provenance"]["sources"] == source
