# Day 20: Q-Learning (Tabular, Gridworld)

## Idea

An agent starts in the top-left corner of a 4x4 grid and has to
reach the goal in the bottom-right corner. Every step costs 0.01,
reaching the goal pays +1, and falling in the pit pays -1 and ends
the episode. The agent keeps a table Q[state, action] and updates
it from experience with the Q-learning rule, so the value of a
good move slowly flows backwards from the goal, one step at a
time, until the greedy policy walks the short route around the
pit. Exploration is epsilon-greedy with epsilon decaying over
2,000 episodes, and everything is seeded, so a fixed seed gives
the same table every time.

## Run

```bash
python algorithms/q-learning/impl.py
pytest algorithms/q-learning/test_q-learning.py
```

## Break it

`test_breaks` trains the same agent with gamma = 0, a myopic
agent that only cares about the next step. It can learn that the
move into the goal pays +1, but that value never flows back to
the start, so from the start every move looks equally bad and
the greedy walk bumps into the wall for all 50 steps instead of
reaching the goal. The same setup with gamma = 0.95 walks the
optimal 6-step route. That is the whole point of the discount:
with no discount there is no planning, only reacting to the
reward right in front of you.

## Workbook

Fork the repo, fill in [workbook.py](workbook.py), and check yourself:

```bash
python algorithms/q-learning/workbook.py
```

Peek at [impl.py](impl.py) only if you get stuck.
