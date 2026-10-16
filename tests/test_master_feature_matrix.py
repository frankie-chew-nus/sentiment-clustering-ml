"""L3-M strict schema, alignment, leakage and export tests."""
import json
import numpy as np
import pandas as pd
import pytest

from master_feature_matrix import (
    EXPECTED_COLUMNS, MASTER_COLUMNS, assemble_frames, assemble_csvs, validate_family,
)

def frames(n=3):
    v = pd.DataFrame({
        "row_id": range(n), "V_pos": [0.2]*n, "V_neg": [0.1]*n,
        "V_neu": [0.7]*n, "V_compound": [0.4]*n,
    })
    neg = pd.DataFrame({
        "row_id": range(n), "N_negation_count": [1]*n,
        "N_intensifier_original_count": [2]*n,
        "N_intensifier_conservative_count": [1]*n,
    })
    s = pd.DataFrame({
        "row_id": range(n), "S_exclamation_count": [3]*n,
        "S_uppercase_ratio": [0.5]*n, "S_emoji_count": [0]*n,
    })
    return v, neg, s

def test_master_schema_and_values():
    parts = frames()
    master = assemble_frames(*parts, expected_rows=3)
    assert master.columns.tolist() == list(MASTER_COLUMNS)
    assert master.row_id.tolist() == [0, 1, 2]
    for part in parts:
        for column in part.columns[1:]:
            assert master[column].equals(part[column])

@pytest.mark.parametrize("family", ["V", "N", "S"])
def test_family_schema(family):
    part = dict(zip(("V", "N", "S"), frames()))[family]
    assert validate_family(part, family, 3).columns.tolist() == list(EXPECTED_COLUMNS[family])

def test_sentiment_column_rejected():
    v, n, s = frames()
    v["airline_sentiment"] = ["negative"]*3
    with pytest.raises(ValueError, match="schema"):
        assemble_frames(v, n, s, 3)

def test_confidence_column_rejected():
    v, n, s = frames()
    n["airline_sentiment_confidence"] = [0.9]*3
    with pytest.raises(ValueError, match="schema"):
        assemble_frames(v, n, s, 3)

@pytest.mark.parametrize("problem", ["duplicate", "missing", "out_of_order", "noninteger"])
def test_bad_row_ids(problem):
    v, n, s = frames()
    if problem == "duplicate": n.loc[1, "row_id"] = 0
    elif problem == "missing": n.loc[1, "row_id"] = 7
    elif problem == "out_of_order": n["row_id"] = [1, 0, 2]
    else: n["row_id"] = [0.0, 1.0, 2.0]
    with pytest.raises(ValueError):
        assemble_frames(v, n, s, 3)

@pytest.mark.parametrize("column,value", [
    ("V_pos", 1.1), ("V_compound", -1.1), ("S_uppercase_ratio", -0.1),
    ("S_exclamation_count", 1.5), ("N_negation_count", -1),
    ("S_emoji_count", np.nan), ("V_neg", np.inf),
])
def test_invalid_values(column, value):
    v, n, s = frames()
    for part in (v, n, s):
        if column in part.columns:
            # A fractional count is deliberately invalid. Convert the test
            # column first so pandas does not warn about incompatible dtype.
            if column == "S_exclamation_count" and value == 1.5:
                part[column] = part[column].astype("float64")
            part.loc[0, column] = value
    with pytest.raises(ValueError):
        assemble_frames(v, n, s, 3)

def test_wrong_row_count():
    v, n, s = frames()
    with pytest.raises(ValueError, match="row count"):
        assemble_frames(v, n, s, 4)

def test_csv_and_manifest(tmp_path):
    v, n, s = frames()
    paths = [tmp_path / f"{name}.csv" for name in ("v", "n", "s")]
    for frame, path in zip((v, n, s), paths):
        frame.to_csv(path, index=False)
    output = tmp_path / "master.csv"
    master, manifest_path = assemble_csvs(*paths, output, expected_rows=3,
        source_sha256={"RAW": "audit-hash", "P1": "p1-hash"})
    manifest = json.loads(manifest_path.read_text())
    assert output.exists()
    assert master.shape == (3, 11)
    assert manifest["row_count"] == 3
    assert manifest["sentiment_labels_loaded"] is False
    assert manifest["scaled"] is False
    assert manifest["source_sha256_reference"]["RAW"] == "audit-hash"
    assert len(manifest["output_sha256"]) == 64
