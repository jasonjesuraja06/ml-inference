"""The README's measured tables must equal the frozen runs, digit for digit.

The claim this repository makes is that every number in a document came from a
committed artifact. That claim is only worth anything if something checks it.
These tests re-derive the README's result tables from `results/<run>/metrics.json`
and fail when the two disagree, so a stale table cannot survive a retrain.

They skip when a run is missing, because a fresh clone has no `results/` until
someone trains. They do not skip when a run is present and its number is wrong.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
README = REPO / "README.md"
RESULTS = REPO / "results"
AGGREGATE_ROWS = {"accuracy", "macro avg", "weighted avg", "micro avg", "samples avg"}

# Row label in the README -> directory under results/
CONFIG_ROWS = {
    "majority class, no training": "majority_baseline",
    "`codebert-base`, plain cross-entropy": "cwe_baseline",
    "`unixcoder-base`, focal + class weights + augmentation": "cwe_improved",
}
# Order of the metric columns in the README's Task A table, after "Train rows".
TASK_A_COLUMNS = ["f1_macro", "f1_weighted", "accuracy", "recall_macro"]
# Order of the per-class F1 columns, after "Holdout support".
PER_CLASS_COLUMNS = ["majority_baseline", "cwe_baseline", "cwe_improved"]


def load(name: str) -> dict | None:
    p = RESULTS / name / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def table_rows(text: str) -> list[list[str]]:
    """Every markdown table row in the document, as stripped cell lists."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def numbers(cell: str) -> list[float]:
    return [float(m) for m in re.findall(r"\d+\.\d+", cell)]


@pytest.fixture(scope="module")
def readme_rows() -> list[list[str]]:
    return table_rows(README.read_text())


@pytest.mark.parametrize(("label", "run"), CONFIG_ROWS.items())
def test_task_a_config_row_matches_its_frozen_run(readme_rows, label, run):
    """Macro F1, weighted F1, accuracy, and macro recall must match results/<run>/."""
    metrics = load(run)
    if metrics is None:
        pytest.skip(f"results/{run}/ not present; run the training first")
    matching = [r for r in readme_rows if r[0] == label]
    assert len(matching) == 1, f"expected exactly one README row for {label!r}, found {len(matching)}"
    cells = matching[0]

    for offset, key in enumerate(TASK_A_COLUMNS, start=2):
        got = numbers(cells[offset])
        assert got, f"{label}: column {offset} ({key}) holds no number: {cells[offset]!r}"
        assert got[0] == pytest.approx(metrics[key], abs=5e-5), (
            f"{label}: README says {key} = {got[0]}, results/{run}/metrics.json says {metrics[key]}"
        )


def test_task_a_per_class_rows_match_the_frozen_runs(readme_rows):
    """Every per-class F1 in the README must equal the frozen per-class report."""
    runs = {name: load(name) for name in PER_CLASS_COLUMNS}
    if any(m is None for m in runs.values()):
        pytest.skip("not every run is present; train them first")
    reference = runs["cwe_improved"]["per_class"]
    classes = [k for k in reference if k not in AGGREGATE_ROWS]
    assert classes, "the improved run reported no per-class scores"

    checked = 0
    for cells in readme_rows:
        if cells[0] not in classes:
            continue
        name = cells[0]
        support = int(cells[1])
        assert support == int(reference[name]["support"]), (
            f"{name}: README support {support} != frozen {int(reference[name]['support'])}"
        )
        for offset, run in enumerate(PER_CLASS_COLUMNS, start=2):
            got = numbers(cells[offset])
            assert got, f"{name}: no number in column {offset}"
            expected = runs[run]["per_class"][name]["f1-score"]
            assert got[0] == pytest.approx(expected, abs=5e-5), (
                f"{name}: README shows {got[0]} for {run}, frozen run says {expected}"
            )
        checked += 1
    assert checked == len(classes), (
        f"README lists {checked} of the {len(classes)} classes; a class table went stale"
    )


def test_task_b_row_matches_the_frozen_devign_run(readme_rows):
    """The measured Devign accuracy and binary F1 must match results/devign_codebert/."""
    metrics = load("devign_codebert")
    if metrics is None:
        pytest.skip("results/devign_codebert/ not present; run `make train-devign` first")
    matching = [r for r in readme_rows if r[0].startswith("**this repository**")]
    assert len(matching) == 1, "expected exactly one measured Devign row in the README"
    cells = matching[0]
    for offset, key in ((1, "accuracy"), (2, "f1_binary")):
        got = numbers(cells[offset])
        assert got, f"no number in Devign column {offset}"
        assert got[0] == pytest.approx(metrics[key], abs=5e-5), (
            f"README Devign {key} = {got[0]}, frozen run says {metrics[key]}"
        )


def test_published_leaderboard_rows_match_the_transcribed_source(readme_rows):
    """The leaderboard rows must match published_baselines.json, not drift from it."""
    published = json.loads((RESULTS / "published_baselines.json").read_text())
    expected = {r["model"]: r["accuracy"] / 100 for r in published["codexglue_defect_detection"]["results"]}
    seen = {}
    for cells in readme_rows:
        if not cells[0].endswith("CodeXGLUE leaderboard"):
            continue
        model = cells[0].rsplit(",", 1)[0].strip()
        seen[model] = numbers(cells[1])[0]
    assert seen, "the README no longer shows the CodeXGLUE leaderboard rows"
    for model, accuracy in seen.items():
        assert model in expected, f"README cites {model!r}, which is not in published_baselines.json"
        assert accuracy == pytest.approx(expected[model], abs=5e-5), (
            f"{model}: README says {accuracy}, published_baselines.json says {expected[model]}"
        )
