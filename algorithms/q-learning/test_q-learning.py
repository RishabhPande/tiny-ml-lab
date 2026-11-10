"""Tests for Q-learning: the trained agent walks the short route
to the goal and avoids the pit, but a myopic agent with gamma = 0
never learns the route, because the goal's reward never flows
back to the start, so every move from the start looks equally bad.
"""

import os
import importlib.util

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "impl_q_learning",
    os.path.join(os.path.dirname(__file__), "impl.py"),
)
_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
GridWorld = _impl.GridWorld
QLearningAgent = _impl.QLearningAgent
train_qlearning = _impl.train_qlearning
rollout = _impl.rollout


def test_works():
    """After training, the greedy policy reaches the goal in the
    optimal 6 steps, never falls in the pit, and training is
    deterministic for a fixed seed."""
    env, agent, score = train_qlearning()

    # Late training is nearly always successful: goal minus steps.
    assert score > 0.8

    # The greedy walk from the start takes the shortest route.
    reached, steps, total, path = rollout(env, agent)
    assert reached
    assert steps == 6
    assert total == 0.95

    # The walk never visits the pit.
    pit_state = env._to_state((2, 2))
    assert pit_state not in path

    # The Q table is real learning, not zeros or garbage.
    assert np.all(np.isfinite(agent.q))
    assert agent.q.max() > 0.9
    start = env._to_state((0, 0))
    assert agent.q[start].max() > 0.5

    # Same seed, same table: training is deterministic.
    _, agent_again, _ = train_qlearning()
    assert np.array_equal(agent.q, agent_again.q)


def test_breaks():
    """A myopic agent (gamma = 0) only values the next step, so the
    goal's reward never flows back to the start: every move from
    the start looks equally bad, the greedy walk bumps into the
    wall for 50 steps, and it never reaches the goal. The failure
    is the discount, not the training: the same setup with
    gamma = 0.95 walks the optimal route."""
    env, trained, _ = train_qlearning()
    _, myopic, _ = train_qlearning(gamma=0.0)

    reached, steps, total, _ = rollout(env, myopic)
    assert not reached
    assert steps == 50
    assert total < 0.0

    # From the start, the myopic Q values are all the same step
    # cost: nothing distinguishes a good move from a bad one.
    start = env._to_state((0, 0))
    assert np.allclose(myopic.q[start], myopic.q[start, 0])
    assert myopic.q[start].max() <= 0.0

    # The same agent with a real discount learns the route.
    reached_t, steps_t, _, _ = rollout(env, trained)
    assert reached_t
    assert steps_t == 6


def test_env_rules():
    """The gridworld itself behaves: walls hold the agent in place,
    the pit ends the episode with -1, and the goal ends it with +1."""
    env = GridWorld()

    # Walking up from the start bumps into the top wall.
    env.reset()
    state, reward, done = env.step(0)
    assert state == env._to_state((0, 0))
    assert reward == -0.01
    assert not done

    # Stepping into the pit ends the episode with -1.
    env.reset()
    env.step(1)  # (0, 1)
    env.step(1)  # (0, 2)
    env.step(2)  # (1, 2)
    state, reward, done = env.step(2)  # (2, 2), the pit
    assert state == env._to_state((2, 2))
    assert reward == -1.0
    assert done

    # The goal and the pit are terminal states.
    assert env.is_terminal(env._to_state((3, 3)))
    assert env.is_terminal(env._to_state((2, 2)))
    assert not env.is_terminal(env._to_state((0, 0)))
