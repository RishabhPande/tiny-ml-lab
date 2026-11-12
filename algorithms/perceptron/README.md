# Day 22: Perceptron

## Idea

The perceptron (Rosenblatt, 1958) is the original linear learner.
It keeps a weight vector and a bias, predicts the sign of the
score, and learns only from mistakes: a misclassified point pulls
the weights toward itself, scaled by its label. If a line exists
that separates the classes, this rule is guaranteed to find one
after finitely many mistakes. If no such line exists, it updates
forever without ever settling, and the mistake counts say so.

## Run

```bash
python algorithms/perceptron/impl.py
pytest algorithms/perceptron/test_perceptron.py
```

## Break it

`test_breaks` trains on XOR, where the positive class sits on
opposite corners and the negative class on the other two. No line
puts the positives on one side, so after 50 epochs the perceptron
has never had a clean sweep and its accuracy stays well under
perfect. Adding one feature, the product x1 * x2, makes the same
points separable and the same rule converges to 100 percent. The
lesson carries through the rest of the repo: logistic regression
and the SVM draw the same straight boundary, and the tiny neural
net only escapes XOR by learning features like that product
instead of being handed them.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/perceptron/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
