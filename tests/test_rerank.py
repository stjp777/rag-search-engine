import pytest

from rag_search import rerank
from rag_search.errors import LLMError
from rag_search.rerank import parse_id_list, parse_score


@pytest.mark.parametrize(
    "text, expected",
    [("8", 8.0), ("Score: 7.5", 7.5), ("8/10", 8.0), ("on a 0-10 scale: 6", 6.0)],
)
def test_parse_score(text, expected):
    assert parse_score(text) == expected


@pytest.mark.parametrize("text", ["no digits", "42"])
def test_parse_score_rejects_bad_replies(text):
    with pytest.raises(ValueError):
        parse_score(text)


def test_parse_id_list_ignores_markdown_fences():
    assert parse_id_list("```json\n[3, 1, 2]\n```") == [3, 1, 2]


def test_parse_id_list_requires_array():
    with pytest.raises(ValueError):
        parse_id_list("none")


RESULTS = [
    {"id": 1, "title": "One", "description": "first"},
    {"id": 2, "title": "Two", "description": "second"},
    {"id": 3, "title": "Three", "description": "third"},
]


def fresh():
    return [dict(r) for r in RESULTS]


def test_batch_rerank_orders_by_model_and_appends_omitted(monkeypatch):
    monkeypatch.setattr(rerank, "ask", lambda prompt, parse: [3, 1])
    assert [r["id"] for r in rerank.rerank_batch("q", fresh())] == [3, 1, 2]


def test_batch_rerank_keeps_order_when_llm_fails(monkeypatch):
    def fail(prompt, parse):
        raise LLMError("down")

    monkeypatch.setattr(rerank, "ask", fail)
    assert [r["id"] for r in rerank.rerank_batch("q", fresh())] == [1, 2, 3]


def test_individual_rerank_sorts_by_score_and_marks_failures(monkeypatch):
    scores = {"first": 2.0, "second": 9.0}

    def fake_ask(prompt, parse):
        for text, score in scores.items():
            if text in prompt:
                return score
        raise LLMError("no score")

    monkeypatch.setattr(rerank, "ask", fake_ask)
    ranked = rerank.rerank_individual("q", fresh())
    assert [r["id"] for r in ranked] == [2, 1, 3]
    assert ranked[-1]["rerank_score"] is None


def test_unknown_method():
    with pytest.raises(ValueError):
        rerank.rerank("q", fresh(), "magic")


def test_individual_rerank_runs_concurrently_and_keeps_scores_aligned(monkeypatch):
    docs = [{"id": i, "title": f"T{i}", "description": f"desc{i}"} for i in range(10)]
    monkeypatch.setattr(rerank, "ask", lambda prompt, parse: float(prompt.split("desc")[1][0]))
    ranked = rerank.rerank_individual("q", docs)
    assert [r["id"] for r in ranked] == list(range(9, -1, -1))
