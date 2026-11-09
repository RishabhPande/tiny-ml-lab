"""Tests for the RAG retriever: keyword queries find the right
document, but a question asked in synonyms the corpus never saw
finds nothing at all."""

import os

import numpy as np

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "impl_rag_retriever",
    os.path.join(os.path.dirname(__file__), "impl.py"),
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
TFIDFRetriever = _impl.TFIDFRetriever
make_corpus = _impl.make_corpus


def test_works():
    """Keyword queries retrieve the right document first, with a
    strong cosine score, and results come back best-first."""
    retriever = TFIDFRetriever(make_corpus())

    hits = retriever.query("stolen credit card fraud detection", k=3)
    assert hits[0][0] == 0
    assert hits[0][1] > 0.5

    hits = retriever.query("plants convert sunlight photosynthesis", k=3)
    assert hits[0][0] == 3
    assert hits[0][1] > 0.5

    hits = retriever.query("quantum qubits superposition", k=3)
    assert hits[0][0] == 6
    assert hits[0][1] > 0.5

    # Scores are sorted best-first and k is respected.
    hits = retriever.query("water", k=3)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)
    assert len(hits) <= 3
    assert all(0.0 < s <= 1.0 for s in scores)

    # The same query twice gives the same answer.
    first = retriever.query("stolen credit card fraud detection", k=3)
    second = retriever.query("stolen credit card fraud detection", k=3)
    assert first == second


def test_breaks():
    """The same fraud question asked in synonyms finds nothing: the
    retriever only knows the words in its corpus, so a vocabulary
    gap looks exactly like no answer existing. The keyword version
    of the same question works, so the failure is the wording, not
    the retriever or the corpus."""
    retriever = TFIDFRetriever(make_corpus())

    hits = retriever.query(
        "crooks swiping plastic to buy goods illegally", k=3
    )
    assert hits == []

    # A query of only stopwords or unknown words is also empty,
    # and it must not produce NaN scores.
    assert retriever.query("the and of to", k=3) == []
    vec = retriever.vectorize("crooks swiping plastic")
    assert np.all(np.isfinite(vec))
    assert not np.any(vec)

    # Control: the same topic in the corpus words retrieves doc 0.
    hits = retriever.query("stolen credit card fraud detection", k=3)
    assert hits[0][0] == 0
    assert hits[0][1] > 0.5


def test_top_k_boundaries():
    """Invalid limits must not become Python slice semantics or vanish on OOV queries."""
    import pytest

    retriever = TFIDFRetriever(make_corpus())
    assert retriever.query("water", k=0) == []
    assert retriever.query("water", k=np.int64(1)) == retriever.query("water", k=1)
    assert len(retriever.query("water", k=100)) == 2
    for text in ("water", "the and of", "unseenword"):
        with pytest.raises(ValueError, match="nonnegative integer"):
            retriever.query(text, k=-1)
        for invalid in (True, False, np.bool_(True), 1.5, "2", None):
            with pytest.raises(TypeError, match="nonnegative integer"):
                retriever.query(text, k=invalid)


def test_empty_corpus_and_empty_documents():
    """No vocabulary is a normal empty result, not a NaN or phantom hit."""
    for docs in ([], ["", "the and of"]):
        retriever = TFIDFRetriever(docs)
        assert retriever.query("water", k=3) == []
        assert np.all(np.isfinite(retriever.matrix))
