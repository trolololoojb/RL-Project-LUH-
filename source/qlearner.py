import numpy as np
from collections import defaultdict

class QLearner:
    # Implements tabular Q-learning with an epsilon-greedy strategy

    def __init__(
        self,
        n_states: int, # number of discrete states in the environment
        n_actions: int, # number of possible actions
        alpha: float = 0.1, # learning rate
        gamma: float = 0.99, # discount factor for future rewards
        seed: int | None = None # random seed for reproducibility
    ):
        # initialize Q-table: each state maps to an array of action values
        self.q = defaultdict(lambda: np.zeros(n_actions, dtype=float))
        self.alpha = alpha          # step size for updates
        self.gamma = gamma          # discount factor for future rewards
        self.n_actions = n_actions
        # random number generator for action selection
        self.rng = np.random.default_rng(seed)

    def act(self, state: int, eps: float) -> int:
        """
        Choose an action based on epsilon-greedy policy.

        If a random draw is below eps, pick a random action;
        otherwise select the action with highest estimated value.
        """
        if self.rng.random() < eps:
            # explore: pick a random action
            return int(self.rng.integers(0, self.n_actions))
        # exploit: choose action with highest Q-value
        return int(np.argmax(self.q[state]))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int
    ) -> None:
        """
        Update the Q-value for the (state, action) pair.

        Uses the standard Q-learning rule:
          Q(s,a) <- Q(s,a) + alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
        """
        # estimate of the best future value
        best_future = np.max(self.q[next_state])
        # temporal-difference error
        td_error = reward + self.gamma * best_future - self.q[state][action]
        # apply the learning rate
        self.q[state][action] += self.alpha * td_error

    def get_q_table(self) -> dict:
        """
        Return the learned Q-table as a regular dictionary.
        """
        return dict(self.q)
