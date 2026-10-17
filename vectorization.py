"""Label-blind, sparse L4-M1 TF-IDF construction.

No clustering, scaling of engineered features, or model selection is performed.
"""
from dataclasses import asdict
import hashlib
import json

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from vectorization_config import REPRESENTATIONS, SPECS

FORBIDDEN = ("airline_sentiment", "sentiment_confidence", "airline_sentiment_confidence")

def _validate_texts(texts, name):
    if isinstance(texts, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of tweet strings")
    values = list(texts)
    if not all(isinstance(x, str) for x in values):
        raise ValueError(f"{name} must contain only strings (no missing values)")
    return values

def build_representation(text_by_variant, representation="word_p1"):
    """Fit TF-IDF on supplied corpus; return CSR matrix and fitted vectorizers.

    Input arrays MUST already be aligned in row_id order. Do not supply labels.
    Each block has L2-normalized rows independently; no final combined-row
    normalization or relative block weighting is applied in M1.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation: {representation}")
    if any(k.lower() in FORBIDDEN or "sentiment_confidence" in k.lower()
           for k in text_by_variant):
        raise ValueError("Sentiment labels/confidence must not enter vectorization")
    required = REPRESENTATIONS[representation]
    arrays = {name: _validate_texts(text_by_variant[name], name)
              for _, name in required}
    if len({len(v) for v in arrays.values()}) != 1:
        raise ValueError("Preprocessing variants have different row counts")
    blocks, fitted, metadata = [], {}, []
    for block_name, input_name in required:
        spec = SPECS[block_name]
        vectorizer = TfidfVectorizer(**asdict(spec), dtype=np.float64)
        matrix = vectorizer.fit_transform(arrays[input_name]).tocsr()
        blocks.append(matrix)
        fitted[block_name] = vectorizer
        metadata.append({"block": block_name, "input": input_name,
                         "features": int(matrix.shape[1]), "nnz": int(matrix.nnz),
                         "parameters": asdict(spec)})
    output = sparse.hstack(blocks, format="csr") if len(blocks) > 1 else blocks[0]
    return output, fitted, metadata

def transform_representation(text_by_variant, representation, fitted):
    """Transform aligned new texts without refitting vocabulary or IDF."""
    required = REPRESENTATIONS[representation]
    arrays = {name: _validate_texts(text_by_variant[name], name)
              for _, name in required}
    if len({len(v) for v in arrays.values()}) != 1:
        raise ValueError("Preprocessing variants have different row counts")
    blocks = [fitted[block].transform(arrays[name]).tocsr()
              for block, name in required]
    return sparse.hstack(blocks, format="csr") if len(blocks) > 1 else blocks[0]

def save_representation(matrix, metadata, output_prefix):
    """Persist sparse matrix and metadata outside GitHub (e.g. Google Drive)."""
    from pathlib import Path
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    matrix_path = str(prefix) + ".npz"
    manifest_path = str(prefix) + ".json"
    sparse.save_npz(matrix_path, matrix)
    digest = hashlib.sha256(Path(matrix_path).read_bytes()).hexdigest()
    manifest = {"shape": list(matrix.shape), "nnz": int(matrix.nnz),
                "matrix_sha256": digest, "blocks": metadata}
    Path(manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return matrix_path, manifest_path
