"""L4-M1 TF-IDF reproducibility, sparse geometry, and leakage tests."""
import numpy as np
import pytest
from scipy import sparse
from vectorization import build_representation, transform_representation, save_representation

@pytest.fixture
def corpus():
    p0 = ["GREAT airline!!!", "BAD airline!!!", "not good airline",
          "very good airline", "bad flight airline", "great flight airline"]
    p1 = [x.lower() for x in p0]
    p2 = [x.replace("airline", "carrier") for x in p1]
    return {"P0": p0, "P1": p1, "P2": p2}

@pytest.mark.parametrize("variant", ["word_p1", "word_p2", "char_p0",
    "char_p1", "char_wb_p0", "combined_p1_p1", "combined_p1_p0"])
def test_sparse_shapes_and_norms(corpus, variant):
    matrix, fitted, meta = build_representation(corpus, variant)
    assert sparse.isspmatrix_csr(matrix)
    assert matrix.shape[0] == 6
    assert matrix.shape[1] == sum(x["features"] for x in meta)
    assert np.isfinite(matrix.data).all()
    transformed = transform_representation(corpus, variant, fitted)
    assert (matrix != transformed).nnz == 0
    for part in meta:
        assert part["parameters"]["norm"] == "l2"

def test_no_label_leakage(corpus):
    corpus["airline_sentiment"] = ["negative"] * 6
    with pytest.raises(ValueError, match="Sentiment"):
        build_representation(corpus)

def test_mismatched_rows(corpus):
    corpus["P0"] = corpus["P0"][:-1]
    with pytest.raises(ValueError, match="row counts"):
        build_representation(corpus, "combined_p1_p0")

def test_missing_values(corpus):
    corpus["P1"][0] = None
    with pytest.raises(ValueError, match="strings"):
        build_representation(corpus)

def test_export(corpus, tmp_path):
    x, _, meta = build_representation(corpus)
    matrix_path, manifest_path = save_representation(x, meta, tmp_path / "word")
    assert sparse.load_npz(matrix_path).shape == x.shape
    assert "matrix_sha256" in open(manifest_path).read()
