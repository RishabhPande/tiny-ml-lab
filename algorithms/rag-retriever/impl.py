"""RAG retriever: TF-IDF cosine retrieval over a tiny synthetic corpus.

Retrieval-augmented generation starts with a retriever: given a
question, find the documents most likely to answer it, then hand
them to the generator. This day builds the retriever half, with
plain TF-IDF vectors and cosine similarity (numpy only). Each
document and each query becomes a bag-of-words vector weighted by
term frequency times inverse document frequency, so rare words
count more than common ones, and the query returns the documents
with the highest cosine similarity to it.

The catch is in the name: it is lexical, not semantic. The
retriever only knows the words in its corpus. Ask in different
words and it finds nothing, even when the right document is
sitting there. That vocabulary gap is the failure mode, and it is
why real RAG systems add embeddings or query expansion on top.
Run: python algorithms/rag-retriever/impl.py
"""

import operator
import re

import numpy as np

STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "for",
    "with", "by", "from", "at", "is", "are", "was", "were", "be",
    "been", "it", "its", "this", "that", "as", "into", "back",
}

DOCS = [
    "Credit card fraud detection uses transaction velocity and device fingerprints to flag stolen cards.",
    "A neural network trained on labeled transactions learns patterns of fraudulent purchases.",
    "The Mars rover collects rock samples and sends photos back to mission control.",
    "Photosynthesis lets plants convert sunlight water and carbon dioxide into energy.",
    "Baking sourdough bread needs flour water salt and a long slow fermentation.",
    "The central bank raised interest rates to slow inflation and cool the housing market.",
    "Quantum computers use qubits and superposition to solve certain problems faster.",
    "Regular exercise and sleep improve memory and lower the risk of heart disease.",
]


def make_corpus():
    """Return the fixed synthetic corpus (8 short documents)."""
    return list(DOCS)


def tokenize(text):
    """Lowercase words, keeping letters only, dropping stopwords."""
    return [t for t in re.findall(r"[a-z]+", text.lower())
            if t not in STOPWORDS]


class TFIDFRetriever:
    """TF-IDF cosine retriever over a fixed list of documents.

    Each document is a TF-IDF vector, L2-normalized, so the cosine
    similarity to a query vector is a plain dot product. A query
    with no vocabulary overlap scores zero on every document and
    returns an empty list instead of a meaningless best guess.
    """

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
        """Turn text into a normalized TF-IDF vector over the vocab.

        Out-of-vocabulary words are ignored; a query with no known
        word at all becomes the zero vector.
        """
        vec = np.zeros(len(self.vocab))
        for t in tokenize(text):
            if t in self.index:
                vec[self.index[t]] += 1.0
        vec = vec * self.idf
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm

    def query(self, text, k=3):
        """Return the top-k (doc index, cosine score) pairs, best first.

        Only documents with a positive score are returned. k must be a
        nonnegative integer (not a boolean); k=0 returns no results.
        Otherwise an empty list means no vocabulary match.
        """
        if isinstance(k, (bool, np.bool_)):
            raise TypeError("k must be a nonnegative integer")
        try:
            k = operator.index(k)
        except TypeError as exc:
            raise TypeError("k must be a nonnegative integer") from exc
        if k < 0:
            raise ValueError("k must be a nonnegative integer")
        q = self.vectorize(text)
        if not np.any(q):
            return []
        scores = self.matrix @ q
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order
                if scores[i] > 0]


if __name__ == "__main__":
    retriever = TFIDFRetriever(make_corpus())
    print(f"RAG retriever over {len(DOCS)} docs, "
          f"vocab size {len(retriever.vocab)} (fixed corpus)")

    # The works case: the query uses the corpus words.
    for question in ["stolen credit card fraud detection",
                     "plants convert sunlight photosynthesis"]:
        print(f"\nquery: {question!r}")
        for idx, score in retriever.query(question, k=3):
            print(f"  [{score:.3f}] doc {idx}: {DOCS[idx]}")

    # The break case: the same fraud question in different words.
    synonym = "crooks swiping plastic to buy goods illegally"
    print(f"\nquery: {synonym!r}")
    hits = retriever.query(synonym, k=3)
    if not hits:
        print("  no documents matched: every word is out of vocabulary,")
        print("  so the fraud document stays hidden even though it")
        print("  answers the question. Lexical retrieval has no synonyms.")
    else:
        for idx, score in hits:
            print(f"  [{score:.3f}] doc {idx}: {DOCS[idx]}")
