import json

import pytest

from rag_search.cli import build_parser, main


def test_help_lists_all_groups(capsys):
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["--help"])
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    for group in ("keyword", "semantic", "hybrid", "generate", "evaluate", "image"):
        assert group in output


def test_keyword_workflow_on_custom_dataset(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    dataset = tmp_path / "books.json"
    dataset.write_text(
        json.dumps(
            [
                {"id": 1, "title": "Dragon Tale", "description": "A dragon guards gold."},
                {"id": 2, "title": "Sea Story", "description": "A sailor crosses the sea."},
            ]
        )
    )
    monkeypatch.setattr("rag_search.config.CACHE_ROOT", tmp_path / "cache")

    assert main(["--dataset", str(dataset), "keyword", "build"]) == 0
    assert main(["--dataset", str(dataset), "keyword", "search", "dragon"]) == 0
    assert "Dragon Tale" in capsys.readouterr().out


def test_missing_index_reports_error_instead_of_traceback(tmp_path, monkeypatch, capsys):
    dataset = tmp_path / "d.json"
    dataset.write_text(json.dumps([{"id": 1, "title": "A", "description": "b"}]))
    monkeypatch.setattr("rag_search.config.CACHE_ROOT", tmp_path / "cache")
    assert main(["--dataset", str(dataset), "keyword", "idf", "bear"]) == 1
    assert "keyword build" in capsys.readouterr().err
