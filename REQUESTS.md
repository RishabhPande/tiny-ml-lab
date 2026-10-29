# Requested Algorithms

Vote for the next algorithm by adding a thumbs-up reaction to its line.
The queue order below is the planned order, but votes move items up: an
algorithm with clearly more votes jumps the queue.

## The queue

- [x] logistic-regression
- [ ] random-forest
- [ ] dbscan
- [ ] gradient-boosting
- [ ] pagerank
- [ ] svm
- [ ] gmm
- [ ] hmm
- [ ] attention
- [ ] lora
- [ ] rag-retriever
- [ ] autoencoder
- [ ] q-learning
- [ ] isolation-forest

## How voting works

React with a thumbs-up on the algorithm you want next. One reaction per
person per algorithm. When an algorithm ships, it moves from this list to the
days table in [README.md](README.md) and to [PROGRESS.md](PROGRESS.md).

## Proposing a new algorithm

Open an issue with the algorithm name, a one-line description, and a
suggested break-it test (the failure mode is the interesting part). If it
fits the rules (one file, numpy only, 60 seconds), it joins the queue.
