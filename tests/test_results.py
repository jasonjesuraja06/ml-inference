"""Tests for the results pipeline: freezing a run, rendering it, and the GPU notebook.

These guard the property the repository depends on for its honesty claim: what a
document says and what the committed artifacts say are the same digits, and the
notebook that reproduces a run calls commands that actually exist.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
RESULTS = REPO / "results"
NOTEBOOK = REPO / "notebooks" / "train_gpu.ipynb"
# Subprocesses inherit the real environment; only the import path is forced, so these
# tests behave the same on a developer machine and on a CI runner.
SUBPROCESS_ENV = {**os.environ, "PYTHONPATH": f"{REPO / 'src'}:{REPO}"}

sys.path.insert(0, str(SCRIPTS))

FAKE_METRICS = {
    "accuracy": 0.5,
    "precision_macro": 0.4,
    "recall_macro": 0.45,
    "f1_macro": 0.42,
    "f1_weighted": 0.48,
    "per_class": {
        "CWE-125": {"precision": 0.6, "recall": 0.5, "f1-score": 0.55, "support": 30.0},
        "__OTHER__": {"precision": 0.3, "recall": 0.4, "f1-score": 0.34, "support": 70.0},
        "accuracy": 0.5,
        "macro avg": {"precision": 0.45, "recall": 0.45, "f1-score": 0.445, "support": 100.0},
        "weighted avg": {"precision": 0.39, "recall": 0.43, "f1-score": 0.40, "support": 100.0},
    },
    "confusion_matrix": [[15, 15], [20, 50]],
    "scope": {"model_name": "microsoft/codebert-base", "epochs": 3, "train_rows": 9854},
    "train_wall_clock_seconds": 1234.5,
}


def test_collect_result_freezes_every_artifact(tmp_path, monkeypatch):
    """A frozen run must carry metrics, command, per-class table, and confusion matrix."""
    import collect_result

    monkeypatch.setattr(collect_result, "REPO_ROOT", tmp_path)
    report = tmp_path / "report.json"
    report.write_text(json.dumps(FAKE_METRICS))
    log = tmp_path / "run.log"
    log.write_text("epoch=0 loss=1.0\n")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_result.py",
            "--name", "demo",
            "--report", str(report),
            "--command", "EPOCHS=3 make train-baseline",
            "--log", str(log),
        ],
    )
    collect_result.main()

    out = tmp_path / "results" / "demo"
    for name in ("metrics.json", "command.txt", "summary.txt", "per_class.md",
                 "confusion_matrix.txt", "stdout.log"):
        assert (out / name).exists(), f"{name} missing from the frozen run"

    assert json.loads((out / "metrics.json").read_text()) == FAKE_METRICS
    assert (out / "command.txt").read_text().strip() == "EPOCHS=3 make train-baseline"
    # The wall clock and the exact command must both survive into the summary.
    summary = (out / "summary.txt").read_text()
    assert "1234.5" in summary and "EPOCHS=3 make train-baseline" in summary


def test_per_class_markdown_lists_classes_and_aggregates_separately():
    """Aggregate rows must not be sorted in among the real classes."""
    import collect_result

    table = collect_result.per_class_markdown(FAKE_METRICS).splitlines()
    body = [row for row in table if row.startswith("| ")][1:]  # drop the header row
    labels = [row.split("|")[1].strip() for row in body]
    # Real classes first, in descending support, then the aggregates.
    assert labels[:2] == ["__OTHER__", "CWE-125"]
    assert set(labels[2:]) == {"macro avg", "weighted avg"}
    assert "0.5500" in collect_result.per_class_markdown(FAKE_METRICS)


def test_confusion_matrix_render_is_square_and_labelled():
    import collect_result

    text = collect_result.confusion_matrix_text(FAKE_METRICS)
    assert "rows = true class, columns = predicted class" in text
    for label in ("CWE-125", "__OTHER__"):
        assert label in text
    # Every count in the matrix must appear; a render that drops cells is worse than none.
    for row in FAKE_METRICS["confusion_matrix"]:
        for cell in row:
            assert str(cell) in text


def test_collapse_progress_bars_keeps_the_last_redraw_and_real_output():
    """A captured tqdm bar must shrink to its last state without losing any real line."""
    import collect_result

    raw = "start\n 0%|  | 0/3\r 33%|# | 1/3\r100%|###| 3/3\n{'loss': 1.5}\ndone\n"
    out = collect_result.collapse_progress_bars(raw)
    assert out.splitlines() == ["start", "100%|###| 3/3", "{'loss': 1.5}", "done"]
    assert out.endswith("\n"), "the trailing newline of the original log must survive"
    assert "0/3" not in out and "1/3" not in out


def test_collapse_progress_bars_is_a_noop_without_carriage_returns():
    import collect_result

    raw = "line one\nline two\n"
    assert collect_result.collapse_progress_bars(raw) == raw


def test_frozen_stdout_log_has_no_carriage_returns(tmp_path, monkeypatch):
    """collect_result must read the log with newline='' or the collapse silently no-ops."""
    import collect_result

    monkeypatch.setattr(collect_result, "REPO_ROOT", tmp_path)
    report = tmp_path / "report.json"
    report.write_text(json.dumps(FAKE_METRICS))
    log = tmp_path / "run.log"
    log.write_bytes(b"train=9\n 0%| | 0/2\r100%|#| 2/2\ndone\n")

    monkeypatch.setattr(
        sys, "argv",
        ["collect_result.py", "--name", "demo", "--report", str(report),
         "--command", "make train-baseline", "--log", str(log)],
    )
    collect_result.main()

    frozen = (tmp_path / "results" / "demo" / "stdout.log").read_bytes()
    assert b"\r" not in frozen
    assert b"0/2" not in frozen, "the intermediate redraw survived; newline handling regressed"
    assert b"100%|#| 2/2" in frozen and b"done" in frozen and b"train=9" in frozen


def test_published_baselines_carry_a_citation_for_every_number():
    """No literature number may sit in the repo without the paper it came from."""
    payload = json.loads((RESULTS / "published_baselines.json").read_text())
    blocks = [v for k, v in payload.items() if isinstance(v, dict) and "results" in v]
    assert blocks, "expected at least one block of published results"
    for block in blocks:
        src = block["source"]
        for field in ("title", "authors", "venue", "url"):
            assert src.get(field), f"{src.get('title')} is missing {field}"
        assert block["results"], "a source with no numbers should not be listed"
        assert isinstance(block["comparable_to_this_repo"], bool)
        assert block["comparability_note"].strip(), "every source must say how it compares"


def test_results_table_runs_against_whatever_is_committed():
    """The table renderer must work on the committed results, not just on fixtures."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "results_table.py")],
        capture_output=True,
        text=True,
        cwd=REPO,
        env=SUBPROCESS_ENV,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Task A: top-10 CWE multiclass" in proc.stdout
    assert "Task B: binary vulnerable/benign" in proc.stdout


@pytest.mark.skipif(not NOTEBOOK.exists(), reason="notebook not present")
class TestGpuNotebook:
    @staticmethod
    def notebook() -> dict:
        return json.loads(NOTEBOOK.read_text())

    def test_ships_with_no_precomputed_output(self):
        """A committed output would be a number nobody ran. Every cell must be clear."""
        for i, cell in enumerate(self.notebook()["cells"]):
            if cell["cell_type"] == "code":
                assert cell["outputs"] == [], f"cell {i} carries precomputed output"
                assert cell["execution_count"] is None, f"cell {i} carries an execution count"

    def test_requests_a_gpu_runtime(self):
        assert self.notebook()["metadata"].get("accelerator") == "GPU"

    def test_every_script_it_calls_exists(self):
        """The notebook calls the repo's scripts by path; a rename must fail here."""
        source = "\n".join(
            "".join(c["source"]) for c in self.notebook()["cells"] if c["cell_type"] == "code"
        )
        called = {
            token
            for token in (t.strip("\"',[]()") for t in source.split())
            if token.startswith("scripts/")
        }
        assert called, "the notebook should drive the repository through its own scripts"
        for path in called:
            assert (REPO / path).exists(), f"notebook calls missing script {path}"

    def test_every_module_it_trains_exists(self):
        source = "\n".join(
            "".join(c["source"]) for c in self.notebook()["cells"] if c["cell_type"] == "code"
        )
        modules = {
            token
            for token in (t.strip("\"',[]()") for t in source.split())
            if token.startswith("ml_inference.train_")
        }
        assert modules, "the notebook should train through the repository's modules"
        for module in modules:
            rel = pathlib.Path("src") / (module.replace(".", "/") + ".py")
            assert (REPO / rel).exists(), f"notebook trains missing module {module}"

    def test_every_flag_it_passes_is_accepted_by_the_cli(self):
        """A flag the notebook passes but the CLI dropped would fail an hour into a GPU rerun."""
        source = "\n".join(
            "".join(c["source"]) for c in self.notebook()["cells"] if c["cell_type"] == "code"
        )
        used = {tok.strip("\"',") for tok in source.split() if tok.strip("\"',").startswith("--")}
        accepted = subprocess.run(
            [sys.executable, str(SCRIPTS / "collect_result.py"), "--help"],
            capture_output=True,
            text=True,
            check=True,
            env=SUBPROCESS_ENV,
        ).stdout
        expected = {"--name", "--report", "--command", "--log"}
        assert expected <= used, f"notebook stopped passing {expected - used} to collect_result.py"
        for flag in expected:
            assert flag in accepted, f"notebook passes {flag}, collect_result.py does not accept it"
