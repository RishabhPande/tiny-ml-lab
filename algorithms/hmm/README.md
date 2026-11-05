# Day 15: Hidden Markov Model (HMM)

## Idea

A hidden Markov model says the observations you see were produced by
a hidden state that changes over time, and each state emits symbols
with its own probabilities. Baum-Welch (EM for HMMs) fits that story:
the E step runs forward-backward to work out, for every time step,
how likely each state was, and the M step refits the start,
transition, and emission probabilities from those occupancies.
Viterbi decoding then recovers the single most likely hidden state
sequence. Scaled forward and backward passes keep the probabilities
from underflowing on long sequences.

## Run

```bash
python algorithms/hmm/impl.py
pytest algorithms/hmm/test_hmm.py
```

## Break it

`test_breaks` gives both hidden states the exact same emission
distribution, so the observations carry no information about which
state produced them. EM still runs and converges, but Viterbi
decoding stays near chance (0.588 on a 0.5 chance baseline). The
optimizer is fine, the model class is the limit: a hidden state that
never shows itself in the observations cannot be recovered, no
matter how long the sequence is. That is the cue to check that your
states actually emit differently before trusting a decoded sequence.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/hmm/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
