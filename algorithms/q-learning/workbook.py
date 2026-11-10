"""Q-learning workbook: fill in the TODOs, then run this file.

Run: python workbook.py
It prints which parts are still TODO. Peek at impl.py only if stuck.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from impl import GridWorld, QLearningAgent  # noqa: E402


class WorkbookAgent(QLearningAgent):
    """TODO: tabular Q-learning with epsilon-greedy exploration.

    The Q table starts at zero. Fill in `update` so the value of
    a good move flows backwards from the goal, one step at a time.
    """

    def update(self, state, action, reward, next_state, done,
               alpha, gamma):
        """TODO: one Q-learning update from a single transition.

        target = reward, plus gamma * max_a' Q[next_state, a'] when
        the episode is not done. Then move Q[state, action] a step
        of size alpha toward the target. Hint: for a terminal
        transition there is no next value, the reward is the whole
        target.
        """
        raise NotImplementedError("TODO: implement update")


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
    env = GridWorld()
    agent = WorkbookAgent(env.n_states, env.n_actions)
    results = [
        _check("update",
               lambda: agent.update(0, 1, -0.01, 1, False, 0.5, 0.95)),
        _check("fit", lambda: agent.fit(env)),
    ]
    if all(results):
        from impl import rollout  # noqa: E402

        reached, steps, total, _ = rollout(env, agent)
        print(f"Greedy rollout: reached goal = {reached}, "
              f"in {steps} steps, total reward {total:.2f}")
        print("All parts implemented. Now run: pytest test_q-learning.py")
    else:
        print("Still TODO: fill in the parts above, then run this file again.")
