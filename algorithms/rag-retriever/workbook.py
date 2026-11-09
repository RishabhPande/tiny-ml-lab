"""RAG retriever workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from impl import make_corpus, tokenize, STOPWORDS  # noqa: E402


class TFIDFRetriever:
    """TODO: a TF-IDF cosine retriever over a fixed list of documents."""

    def __init__(self, docs):
        self.docs = list(docs)
        tokenized = [tokenize(d) for d in self.docs]
        self.vocab = sorted({t for toks in tokenized for t in toks})
        self.index = {t: i for i, t in enumerate(self.vocab)}
        n_docs = len(self.docs)
        df = np.zeros(len(self.vocab))
        for toks in tokenized:
            for t in set(toks):
                df[self.index[t]] += 1.0
        self.idf = np.log((n_docs + 1.0) / (df + 1.0)) + 1.0
        mat = np.zeros((n_docs, len(self.vocab)))
        for d, toks in enumerate(tokenized):
            for t in toks:
                mat[d, self.index[t]] += 1.0
        mat = mat * self.idf
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self.matrix = mat / np.where(norms == 0, 1.0, norms)

    def vectorize(self, text):
        """TODO: turn text into a normalized TF-IDF vector over the
        vocab. Count each known token, multiply by idf, L2-normalize.
        Out-of-vocabulary words are ignored; a query with no known
        word at all returns the zero vector."""
        raise NotImplementedError("TODO: implement vectorize")

    def query(self, text, k=3):
        """TODO: return the top-k (doc index, cosine score) pairs,
        best first, only with positive scores. Validate k as a nonnegative
        integer (not a boolean), including for empty/OOV queries; k=0
        returns no results. An empty list otherwise means
        the query matched nothing in the corpus. Hint: cosine
        similarity here is a dot product, because the document
        rows and the query vector are already normalized."""
        raise NotImplementedError("TODO: implement query")


def _check(name, fn):
    try:
        fn()
    except NotImplementedError:
        print(f"[TODO] {name} is not implemented yet")
        return False
    except Exception as exc:  # noqa: BLE001 - workbook feedback, not a test
        print(f"[ERROR] {name} raised {type(exc).__name__}: {exc}")
        return False
    print(f"[OK] {name} runs")
    return True


if __name__ == "__main__":
    retriever = TFIDFRetriever(make_corpus())
    results = [
        _check("vectorize",
               lambda: retriever.vectorize("stolen credit card fraud")),
        _check("query",
               lambda: retriever.query("stolen credit card fraud", k=3)),
    ]
    if all(results):
        print("All parts implemented. Now run: pytest test_rag_retriever.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
