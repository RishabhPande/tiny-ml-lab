import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from unittest import mock

import pytest

spec = importlib.util.spec_from_file_location("workbook_checker", Path(__file__).resolve().parents[1] / "scripts/check_workbooks.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


@pytest.mark.parametrize("output,code,status", [
    ("[TODO] fit is not implemented yet\nStill TODO", 0, "todo"),
    ("[ERROR] fit raised ValueError\n[TODO] predict", 0, "error"),
    ("[OK] fit runs\nAll parts implemented. Now run: pytest", 0, "self-check-ok"),
    ("[OK] fit runs\nAll parts implemented.", 1, "error"),
    ("", 0, "unrecognized"),
    ("[OK] fit runs", 0, "unrecognized"),
    ("All parts implemented.", 0, "unrecognized"),
])
def test_status_is_not_inferred_from_zero_exit(output, code, status, tmp_path):
    path = tmp_path / "example/workbook.py"
    path.parent.mkdir()
    path.write_text("print(" + repr(output) + ")\nraise SystemExit(" + str(code) + ")\n")
    assert checker.check_workbook(path, 2)["status"] == status


def test_timeout_and_os_error(tmp_path):
    path = tmp_path / "workbook.py"
    with mock.patch.object(checker.subprocess, "run", side_effect=subprocess.TimeoutExpired("python", 1)):
        assert checker.check_workbook(path, 1)["status"] == "timeout"
    with mock.patch.object(checker.subprocess, "run", side_effect=OSError):
        assert checker.check_workbook(path, 1)["status"] == "error"


@pytest.fixture
def workbooks(tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    for name, body in (("a", "[TODO] fit"), ("b", "[OK] fit runs\nAll parts implemented.")):
        path = tmp_path / "algorithms" / name / "workbook.py"
        path.parent.mkdir(parents=True)
        path.write_text("print(" + repr(body) + ")")
    return tmp_path


def test_selected_duplicate_names_are_run_once_and_json_is_scoped(workbooks, capsys):
    assert checker.main(["b", "b", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["results"] == [{"algorithm": "b", "status": "self-check-ok"}]
    assert "not correctness" in data["scope"]


def test_all_incomplete_exit_one_without_changing_files(workbooks, capsys):
    progress = workbooks / "PROGRESS.md"
    progress.write_text("- [ ] a\n")
    before = {p: p.read_bytes() for p in workbooks.rglob("*") if p.is_file()}
    assert checker.main([]) == 1
    out = capsys.readouterr().out
    assert "todo" in out and "self-check-ok" in out
    assert all(p.read_bytes() == data for p, data in before.items())


@pytest.mark.parametrize("args", [["../a"], ["missing"], ["--timeout", "0"], ["--timeout", "nan"], ["--timeout", "inf"]])
def test_invalid_selection_or_timeout_is_usage_error(workbooks, args):
    with pytest.raises(SystemExit) as error:
        checker.main(args)
    assert error.value.code == 2


def test_no_workbooks_is_error(tmp_path, monkeypatch):
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    with pytest.raises(SystemExit) as error:
        checker.main([])
    assert error.value.code == 2


def test_each_workbook_runs_in_a_fresh_process_and_its_own_directory(tmp_path):
    path = tmp_path / "example/workbook.py"
    path.parent.mkdir()
    (path.parent / "fixture.txt").write_text("[OK] fit runs\nAll parts implemented.")
    path.write_text("from pathlib import Path\nprint(Path('fixture.txt').read_text())")
    assert checker.check_workbook(path, 2)["status"] == "self-check-ok"
