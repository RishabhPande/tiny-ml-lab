# Day 18: RAG Retriever (TF-IDF Cosine Retrieval)

## Idea

Retrieval-augmented generation starts with a retriever: given a
question, find the documents most likely to answer it, then hand
them to the generator. This day builds the retriever half with
plain TF-IDF vectors and cosine similarity. Each document and each
query becomes a bag-of-words vector weighted by term frequency
times inverse document frequency, so rare words count more than
common ones, and the query returns the documents with the highest
cosine similarity. The corpus is eight short synthetic documents
on different topics, from fraud detection to photosynthesis.

## Run

```bash
python algorithms/rag-retriever/impl.py
pytest algorithms/rag-retriever/test_rag_retriever.py
```

## Break it

`test_breaks` asks the fraud question in different words:
"crooks swiping plastic to buy goods illegally". It returns
nothing, even though the fraud document answers it, because the
retriever only knows the words in its corpus and none of those
words appear there. The same question in the corpus words
retrieves the fraud document with a cosine score above 0.5, so
the failure is the wording, not the retriever or the corpus.
That vocabulary gap is why real RAG systems add embeddings or
query expansion on top of lexical retrieval: matching words is
not the same as matching meaning.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/rag-retriever/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
