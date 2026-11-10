"""Q-Learning: tabular value learning on a small gridworld.

An agent starts in the top-left corner of a 4x4 grid and has to
reach the goal in the bottom-right corner. Every step costs a
little (-0.01), reaching the goal pays +1, and falling in the pit
pays -1 and ends the episode. The agent knows nothing at first:
it keeps a table Q[state, action] and updates it from experience
with the Q-learning rule

    Q[s, a] += alpha * (r + gamma * max_a' Q[s', a'] - Q[s, a])

so the value of a good move slowly flows backwards from the goal,
one step at a time, until the greedy policy walks the short route
around the pit. Exploration is epsilon-greedy with epsilon decaying
over episodes, and everything is seeded, so a fixed seed gives the
same table every time.

The break-it test trains the same agent with gamma = 0, a myopic
agent that only cares about the next step. It can learn that the
move into the goal pays +1, but that value never flows back to the
start, so from the start every move looks equally bad and the
agent bumps into the wall forever instead of walking to the goal.

Run: python algorithms/q-learning/impl.py
"""

import numpy as np

ACTIONS = ["up", "right", "down", "left"]
_DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


class GridWorld:
    """A small deterministic gridworld with a goal and a pit.

    Rows and columns are numbered from the top-left corner. The
    agent starts at `start`, the goal pays +1 and ends the
    episode, each pit pays -1 and ends the episode, and every
    other step pays `step_reward`. Walking into a wall leaves
    the agent where it is. States are numbered row-major, so
    state = row * n_cols + col.
    """

    def __init__(self, n_rows=4, n_cols=4, start=(0, 0), goal=(3, 3),
                 pits=((2, 2),), step_reward=-0.01):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.start = start
        self.goal = goal
        self.pits = tuple(pits)
        self.step_reward = step_reward
        self.n_states = n_rows * n_cols
        self.n_actions = 4
        self.state = self._to_state(start)

    def _to_state(self, pos):
        return pos[0] * self.n_cols + pos[1]

    def _to_pos(self, state):
        return (state // self.n_cols, state % self.n_cols)

    def reset(self):
        """Put the agent back at the start and return its state."""
        self.state = self._to_state(self.start)
        return self.state

    def step(self, action):
        """Take one step. Returns (next_state, reward, done)."""
        row, col = self._to_pos(self.state)
        d_row, d_col = _DELTAS[action]
        row = min(max(row + d_row, 0), self.n_rows - 1)
        col = min(max(col + d_col, 0), self.n_cols - 1)
        self.state = self._to_state((row, col))
        if (row, col) == self.goal:
            return self.state, 1.0, True
        if (row, col) in self.pits:
            return self.state, -1.0, True
        return self.state, self.step_reward, False

    def is_terminal(self, state):
        """True for the goal state and every pit state."""
        pos = self._to_pos(state)
        return pos == self.goal or pos in self.pits


class QLearningAgent:
    """Tabular Q-learning with epsilon-greedy exploration.

    The Q table starts at zero and is updated in place during
    `fit`. The greedy action is the argmax of the row, with ties
    broken by the lowest action index, so greedy play is fully
    deterministic once training is done.
    """

    def __init__(self, n_states, n_actions=4):
        self.q = np.zeros((n_states, n_actions))

    def greedy_action(self, state):
        """Best action in this state by current Q values."""
        return int(np.argmax(self.q[state]))

    def act(self, state, epsilon, rng):
        """Epsilon-greedy action: random with prob epsilon."""
        if rng.random() < epsilon:
            return int(rng.integers(self.q.shape[1]))
        return self.greedy_action(state)

    def update(self, state, action, reward, next_state, done,
               alpha, gamma):
        """One Q-learning update from a single transition."""
        target = reward
        if not done:
            target += gamma * float(np.max(self.q[next_state]))
        self.q[state, action] += alpha * (target - self.q[state, action])

    def fit(self, env, episodes=2000, alpha=0.5, gamma=0.95,
            epsilon_start=1.0, epsilon_end=0.05, seed=0,
            max_steps=100):
        """Train on the env. Returns the mean reward of the last
        100 episodes. Epsilon decays linearly from epsilon_start
        to epsilon_end across the episodes."""
        rng = np.random.default_rng(seed)
        recent = []
        for episode in range(episodes):
            frac = episode / max(episodes - 1, 1)
            epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)
            state = env.reset()
            total = 0.0
            for _ in range(max_steps):
                action = self.act(state, epsilon, rng)
                next_state, reward, done = env.step(action)
                self.update(state, action, reward, next_state, done,
                            alpha, gamma)
                total += reward
                state = next_state
                if done:
                    break
            recent.append(total)
        return float(np.mean(recent[-100:]))

    def greedy_policy(self):
        """Best action per state as an array of action indices."""
        return np.argmax(self.q, axis=1)


def train_qlearning(episodes=2000, alpha=0.5, gamma=0.95, seed=0):
    """Build a gridworld, train an agent, return (env, agent, score)."""
    env = GridWorld()
    agent = QLearningAgent(env.n_states, env.n_actions)
    score = agent.fit(env, episodes=episodes, alpha=alpha,
                      gamma=gamma, seed=seed)
    return env, agent, score


def rollout(env, agent, max_steps=50):
    """Run the greedy policy once. Returns (reached_goal, steps,
    total_reward, path) where path lists the visited states."""
    state = env.reset()
    path = [state]
    total = 0.0
    for step in range(1, max_steps + 1):
        action = agent.greedy_action(state)
        state, reward, done = env.step(action)
        path.append(state)
        total += reward
        if done:
            return state == env._to_state(env.goal), step, total, path
    return False, max_steps, total, path


def render_policy(env, agent):
    """The greedy policy as a grid of arrows, G for goal, X for pit."""
    arrows = {"up": "^", "right": ">", "down": "v", "left": "<"}
    policy = agent.greedy_policy()
    rows = []
    for row in range(env.n_rows):
        cells = []
        for col in range(env.n_cols):
            pos = (row, col)
            if pos == env.goal:
                cells.append("G")
            elif pos in env.pits:
                cells.append("X")
            else:
                cells.append(arrows[ACTIONS[int(policy[env._to_state(pos)])]])
        rows.append(" ".join(cells))
    return "\n".join(rows)


if __name__ == "__main__":
    env, agent, score = train_qlearning()
    print("Q-learning on a 4x4 gridworld (fixed seed)")
    print(f"Goal (3, 3) pays +1, pit (2, 2) pays -1, "
          f"each step costs 0.01")
    print(f"Mean reward over the last 100 training episodes: "
          f"{score:.3f}")
    print("\nLearned greedy policy (G = goal, X = pit):")
    print(render_policy(env, agent))

    reached, steps, total, _ = rollout(env, agent)
    print(f"\nGreedy rollout from the start: reached goal = {reached}, "
          f"in {steps} steps, total reward {total:.2f}")

    # The break case: a myopic agent with gamma = 0 only values the
    # next step, so the goal's reward never flows back to the start.
    _, myopic, _ = train_qlearning(gamma=0.0)
    reached_m, steps_m, total_m, _ = rollout(env, myopic)
    print(f"\nMyopic agent (gamma = 0): reached goal = {reached_m}, "
          f"in {steps_m} steps, total reward {total_m:.2f}")
    print("It learned the move into the goal pays, but from the start")
    print("every move looks equally bad, so it bumps into the wall")
    print("instead of walking to the goal.")
