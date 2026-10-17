"""L4-M4 small-corpus FastText and pooling validation tests."""
import json
import numpy as np
import pytest
from fasttext_representation import FastTextSpec, FastTextTweetEncoder, save_fasttext

@pytest.fixture
def corpus():
    return [
        "excellent airline flight", "terrible airline flight",
        "not good flight", "very good flight", "excellent service airline",
        "terrible service airline", "good airline service", "bad airline service",
        "delayed flight service", "thank you airline", "flight was delayed",
        "airline flight was good",
    ]

@pytest.fixture
def encoder():
    return FastTextTweetEncoder(FastTextSpec(vector_size=12, epochs=2, min_count=1))

def test_shapes_and_norms(corpus, encoder):
    x, zeros = encoder.fit_transform(corpus)
    assert x.shape == (len(corpus), 12)
    assert np.isfinite(x).all()
    norms = np.linalg.norm(x, axis=1)
    assert np.allclose(norms[norms > 0], 1, atol=1e-12)
    assert not zeros

def test_roundtrip_transform(corpus, encoder):
    a, _ = encoder.fit_transform(corpus)
    b, _ = encoder.transform(corpus)
    np.testing.assert_allclose(a, b, atol=1e-12)

def test_empty_and_unseen(corpus, encoder):
    encoder.fit(corpus)
    x, zeros = encoder.transform(["", "completely unseen vocabulary"])
    assert x.shape == (2, 12)
    assert zeros == [0, 1]
    assert np.all(x == 0)

def test_missing_text(corpus, encoder):
    corpus[0] = None
    with pytest.raises(ValueError, match="strings"):
        encoder.fit(corpus)

def test_fit_required(encoder):
    with pytest.raises(RuntimeError, match="Fit"):
        encoder.transform(["good flight"])

def test_bad_spec():
    with pytest.raises(ValueError, match="Invalid"):
        FastTextSpec(vector_size=0)

def test_manifest(corpus, encoder):
    encoder.fit(corpus)
    meta = encoder.manifest()
    assert meta["spec"]["sg"] == 1
    assert meta["tfidf_vocabulary_size"] > 0

def test_export(corpus, encoder, tmp_path):
    x, zeros = encoder.fit_transform(corpus)
    p, j = save_fasttext(x, zeros, encoder.manifest(), tmp_path / "fasttext",
                         {"git_revision": "test", "p1_sha256": "a" * 64})
    loaded = np.load(p)["embeddings"]
    np.testing.assert_array_equal(loaded, x)
    assert json.load(open(j))["matrix_sha256"]

def test_export_requires_provenance(corpus, encoder, tmp_path):
    x, zeros = encoder.fit_transform(corpus)
    with pytest.raises(ValueError, match="git_revision"):
        save_fasttext(x, zeros, encoder.manifest(), tmp_path / "x", {})

def test_blank_corpus(encoder):
    with pytest.raises(ValueError, match="empty"):
        encoder.fit(["", ""])
