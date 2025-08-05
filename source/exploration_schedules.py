import numpy as np

# Provides constant and EZ-greedy exploration strategies

def fixed_eps_schedule(eps: float):
    """
    Create a function returning a fixed exploration rate.
    """
    return lambda episode, t_step: eps

class EZGreedy:
    """
    Keeps an exploratory action active for a fixed number of steps.
    """
    def __init__(self, base_eps: float, k: int, rng: np.random.Generator):
        # base exploration probability
        self.base_eps = base_eps
        # number of steps to repeat a chosen action
        self.k = k # repetition phase length
        self.rng = rng
        self.steps_left = 0
        self.last_action = None

    def select_action(self, q_values: np.ndarray) -> int:
        # if in repetition phase, use previous action
        if self.steps_left > 0:
            self.steps_left -= 1
            return self.last_action  # type: ignore

        # decide whether to explore new action
        if self.rng.random() < self.base_eps:
            action = int(self.rng.integers(0, q_values.shape[0]))
            self.last_action = action
            self.steps_left = self.k - 1
            return action

        # choose action with highest Q-value
        return int(np.argmax(q_values))
