"""L3-M: assemble validated V/N/S feature CSVs without sentiment labels.

The three inputs must share the same row_id alignment and original dataset order.
No scaling, model fitting, or sentiment-label access occurs here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from feature_engineering_config import VADER_COLUMNS, N_COLUMNS, S_COLUMNS

EXPECTED_COLUMNS = {
    "V": ("row_id", *VADER_COLUMNS),
    "N": ("row_id", *N_COLUMNS),
    "S": ("row_id", *S_COLUMNS),
}
MASTER_COLUMNS = ("row_id", *VADER_COLUMNS, *N_COLUMNS, *S_COLUMNS)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_family(frame: pd.DataFrame, family: str, expected_rows: int = 14639) -> pd.DataFrame:
    """Fail closed on schema, row alignment, numeric integrity, and value ranges."""
    if family not in EXPECTED_COLUMNS:
        raise ValueError(f"Unknown family: {family}")
    expected = EXPECTED_COLUMNS[family]
    if set(frame.columns) != set(expected) or len(frame.columns) != len(expected):
        raise ValueError(f"{family} schema mismatch: expected {expected}, found {tuple(frame.columns)}")
    if len(frame) != expected_rows:
        raise ValueError(f"{family} row count {len(frame)} != {expected_rows}")
    if frame["row_id"].isna().any() or frame["row_id"].duplicated().any():
        raise ValueError(f"{family} missing or duplicate row IDs")
    if not pd.api.types.is_integer_dtype(frame["row_id"]):
        raise ValueError(f"{family} row IDs must be integers")
    if frame["row_id"].tolist() != list(range(expected_rows)):
        raise ValueError(f"{family} row IDs must be sequential in source order")
    result = frame.loc[:, list(expected)].copy()
    for column in expected[1:]:
        if not pd.api.types.is_numeric_dtype(result[column]):
            raise ValueError(f"{family}.{column} must be numeric")
        values = result[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{family}.{column} contains missing or non-finite values")
        if column == "V_compound":
            valid = (values >= -1) & (values <= 1)
        elif column.startswith("V_") or column == "S_uppercase_ratio":
            valid = (values >= 0) & (values <= 1)
        else:
            valid = (values >= 0) & (values == np.floor(values))
        if not valid.all():
            raise ValueError(f"{family}.{column} contains out-of-range values")
    return result


def assemble_frames(vader: pd.DataFrame, negation: pd.DataFrame,
                    structural: pd.DataFrame, expected_rows: int = 14639) -> pd.DataFrame:
    """Join validated family blocks by row_id without changing any feature value."""
    blocks = [
        validate_family(vader, "V", expected_rows),
        validate_family(negation, "N", expected_rows),
        validate_family(structural, "S", expected_rows),
    ]
    master = blocks[0]
    for block in blocks[1:]:
        master = master.merge(block, on="row_id", how="inner", validate="one_to_one", sort=False)
    if master.columns.tolist() != list(MASTER_COLUMNS):
        raise AssertionError("Unexpected master column order")
    if len(master) != expected_rows or master["row_id"].tolist() != list(range(expected_rows)):
        raise AssertionError("Master row alignment changed")
    for block in blocks:
        for column in block.columns[1:]:
            if not master[column].equals(block[column]):
                raise AssertionError(f"Feature values changed during assembly: {column}")
    return master


def assemble_csvs(vader_path: str | Path, negation_path: str | Path,
                  structural_path: str | Path, output_path: str | Path,
                  expected_rows: int = 14639, source_sha256: dict[str, str] | None = None):
    """Export master CSV and adjacent manifest; never load sentiment labels."""
    paths = {"V": Path(vader_path), "N": Path(negation_path), "S": Path(structural_path)}
    frames = {family: pd.read_csv(path) for family, path in paths.items()}
    master = assemble_frames(frames["V"], frames["N"], frames["S"], expected_rows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(output, index=False)
    manifest = {
        "schema_version": "L3-M-v1",
        "row_count": len(master),
        "columns": list(MASTER_COLUMNS),
        "input_files": {k: {"name": v.name, "sha256": sha256_file(v)}
                        for k, v in paths.items()},
        "source_sha256_reference": source_sha256 or {},
        "output_file": output.name,
        "output_sha256": sha256_file(output),
        "sentiment_labels_loaded": False,
        "scaled": False,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return master, manifest_path
