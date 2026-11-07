# Day 17: LoRA (Low-Rank Adaptation)

## Idea

Fine-tuning usually updates every weight. LoRA (Hu et al., 2021)
freezes the pretrained weight W0 and learns only a low-rank update
dW = B @ A, with A of shape (r, d_in) and B of shape (d_out, r),
where r is far smaller than either dimension. A starts random and
B starts at zero, so the adapter is silent at first: the model
behaves exactly like the frozen layer until training moves it.
Only A and B get gradients, so a 16 to 8 layer trains 96 parameters
instead of 128, and the saving grows with model size. After
training, the adapter merges back into W0, so inference costs
nothing extra.

## Run

```bash
python algorithms/lora/impl.py
pytest algorithms/lora/test_lora.py
```

## Break it

`test_breaks` makes the true update a random full-rank matrix and
trains a rank-1 adapter on it. The loss stalls with about 70% of
the frozen loss still there, because a rank-1 product can only
point along one direction and the task needs eight. The same task
with a rank-8 adapter fits, so the failure is the rank bottleneck,
not the training. That is the rule for picking r in practice: it
has to cover the rank of the change the new task actually needs,
and no amount of extra steps fixes an adapter that is too narrow.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/lora/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
