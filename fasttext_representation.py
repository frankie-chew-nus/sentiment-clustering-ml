"""L4-M4 optional, label-blind corpus-trained FastText tweet representation."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

@dataclass(frozen=True)
class FastTextSpec:
    vector_size: int = 100
    window: int = 5
    min_count: int = 2
    epochs: int = 10
    sg: int = 1
    seed: int = 5001
    workers: int = 1
    min_n: int = 3
    max_n: int = 6

    def __post_init__(self):
        if min(self.vector_size, self.window, self.min_count, self.epochs,
               self.workers, self.min_n) < 1 or self.max_n < self.min_n:
            raise ValueError("Invalid FastText configuration")

def _tokens(texts):
    if isinstance(texts, (str, bytes)):
        raise TypeError("Expected aligned sequence of processed P1 tweet strings")
    values = list(texts)
    if not all(isinstance(t, str) for t in values):
        raise ValueError("Text must contain only strings")
    return [t.split() for t in values]

class FastTextTweetEncoder:
    """Train once on P1 corpus, fit unigram TF-IDF, then pool token embeddings.

    Out-of-explicit-vocabulary tokens can still receive FastText subword vectors.
    A zero pooled vector remains zero; its row index is recorded for audit.
    """
    def __init__(self, spec=None):
        self.spec = spec or FastTextSpec()
        self.model_ = None
        self.tfidf_ = None
        self.idf_ = None

    def fit(self, texts):
        from gensim.models import FastText
        tokens = _tokens(texts)
        if not any(tokens):
            raise ValueError("Cannot fit an empty token corpus")
        self.model_ = FastText(
            sentences=tokens, vector_size=self.spec.vector_size,
            window=self.spec.window, min_count=self.spec.min_count,
            epochs=self.spec.epochs, sg=self.spec.sg, seed=self.spec.seed,
            workers=self.spec.workers, min_n=self.spec.min_n,
            max_n=self.spec.max_n
        )
        # Use exactly the same whitespace tokenization for both models.
        self.tfidf_ = TfidfVectorizer(
            analyzer=lambda text: text.split(), lowercase=False,
            token_pattern=None, norm=None, dtype=np.float64
        )
        self.tfidf_.fit([" ".join(t) for t in tokens])
        self.idf_ = dict(zip(self.tfidf_.get_feature_names_out(),
                             self.tfidf_.idf_))
        return self

    def transform(self, texts):
        if self.model_ is None:
            raise RuntimeError("Fit the encoder before transform")
        tokens = _tokens(texts)
        x = np.zeros((len(tokens), self.spec.vector_size), dtype=np.float64)
        zero_rows = []
        for i, row in enumerate(tokens):
            weighted_sum = np.zeros(self.spec.vector_size, dtype=np.float64)
            total_weight = 0.0
            for token in row:
                weight = self.idf_.get(token)
                if weight is None:
                    continue
                try:
                    vector = self.model_.wv[token]
                except KeyError:
                    continue
                weighted_sum += weight * vector
                total_weight += weight
            if total_weight > 0:
                pooled = weighted_sum / total_weight
                norm = np.linalg.norm(pooled)
                if norm > 0:
                    x[i] = pooled / norm
            if not np.any(x[i]):
                zero_rows.append(i)
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite pooled embeddings")
        return x, zero_rows

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)

    def manifest(self):
        if self.model_ is None:
            raise RuntimeError("Encoder not fitted")
        return {"model": "corpus_trained_fasttext",
                "pooling": "tfidf_idf_weighted_mean_then_l2",
                "tokenization": "P1 whitespace split",
                "spec": asdict(self.spec),
                "explicit_vocabulary_size": len(self.model_.wv),
                "tfidf_vocabulary_size": len(self.idf_)}

def save_fasttext(matrix, zero_rows, manifest, prefix, provenance):
    """Save dense 100-D matrix and a manifest with required source provenance."""
    if not provenance or not provenance.get("git_revision") or not provenance.get("p1_sha256"):
        raise ValueError("git_revision and p1_sha256 required")
    sha = provenance["p1_sha256"]
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha.lower()):
        raise ValueError("Invalid P1 SHA256")
    path = Path(str(prefix) + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=np.asarray(matrix, dtype=np.float64))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    meta = {**manifest, "shape": list(matrix.shape), "zero_rows": zero_rows,
            "matrix_sha256": digest, "provenance": provenance}
    json_path = Path(str(prefix) + ".json")
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return str(path), str(json_path)
