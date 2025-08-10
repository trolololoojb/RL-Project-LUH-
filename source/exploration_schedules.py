import numpy as np

# Provides annealed, constant and EZ-greedy exploration strategies

def annealed_linear(eps_start: float, eps_end: float, horizon: int, n_episodes: int):
    """
    Linear annealing from eps_start -> eps_end over total training steps.
    Usage: eps_fn = annealed_linear(1.0, 0.05, horizon=500, n_episodes=2000)
           eps = eps_fn(ep, t_step)
    """
    total_steps = max(1, horizon * n_episodes)
    def fn(episode: int, t_step: int) -> float:
        # Calculate the current step in the total training process
        g = episode * horizon + t_step
        frac = min(max(g / total_steps, 0.0), 1.0)
        return eps_start + (eps_end - eps_start) * frac
    return fn

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
            action = int(self.rng.integers(0, q_values.shape[0])) # random action
            self.last_action = action
            self.steps_left = self.k - 1
            return action

        # choose action with highest Q-value
        return int(np.argmax(q_values))
