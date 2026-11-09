import json

import pytest

from rag_search.dataset import DatasetError, cache_dir_for, load_dataset


def write(tmp_path, payload):
    path = tmp_path / "docs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_plain_list(tmp_path):
    path = write(tmp_path, [{"id": 1, "title": "A", "description": "x"}])
    assert load_dataset(path)[0]["title"] == "A"


def test_loads_wrapped_list(tmp_path):
    path = write(tmp_path, {"movies": [{"id": "a", "title": "A", "description": "x"}]})
    assert load_dataset(path)[0]["id"] == "a"


@pytest.mark.parametrize(
    "payload",
    [
        [],
        [{"id": 1, "title": "A"}],
        [{"id": 1, "title": "A", "description": "x"}] * 2,
        {"unrelated": 1},
    ],
)
def test_invalid_datasets_are_rejected(tmp_path, payload):
    with pytest.raises(DatasetError):
        load_dataset(write(tmp_path, payload))


def test_missing_file():
    with pytest.raises(DatasetError):
        load_dataset("does-not-exist.json")


def test_cache_dir_changes_with_contents(tmp_path):
    path = write(tmp_path, [{"id": 1, "title": "A", "description": "x"}])
    before = cache_dir_for(path)
    write(tmp_path, [{"id": 1, "title": "A", "description": "changed"}])
    assert cache_dir_for(path) != before


def test_parse_filters():
    from rag_search.dataset import parse_filters

    assert parse_filters(["genre=Comedy", " year = 1999 "]) == {"genre": "Comedy", "year": "1999"}
    with pytest.raises(ValueError):
        parse_filters(["genre"])


def test_matches_filters():
    from rag_search.dataset import matches_filters

    doc = {"genre": ["Comedy", "Family"], "year": 1999}
    assert matches_filters(doc, {"genre": "comedy", "year": "1999"})
    assert not matches_filters(doc, {"genre": "Horror"})
    assert not matches_filters(doc, {"rating": "PG"})
    assert matches_filters(doc, None)
