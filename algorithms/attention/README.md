# Day 16: Attention (Scaled Dot-Product Self-Attention)

## Idea

Attention lets every token look at every other token and decide how
much each one matters. Each token is projected into a query, a key,
and a value; the weight from token i to token j is the softmax of
the query-key dot product, scaled by one over the square root of the
model width so the scores stay in a range where softmax can still
tell them apart. Token i's output is then the weighted mix of all
the values, passed through an output projection. This single head is
the core of the Transformer: stack it, add positions, and you get
the architecture behind modern language models.

## Run

```bash
python algorithms/attention/impl.py
pytest algorithms/attention/test_attention.py
```

## Break it

`test_breaks` feeds in six identical tokens. Every key matches
every query equally, so the softmax goes perfectly uniform (1/6
each) and every token's output is the same mean value. Attention
says nothing about which token is which, because the tokens carry
nothing to tell apart. That is the same reason plain attention is
permutation-equivariant: shuffle the input rows and the outputs
shuffle the same way, with no trace of the original order. It is
why Transformers need positional encodings added on top: without
them, attention alone cannot see order at all.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/attention/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
