import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from dataclasses import dataclass
from typing import Optional

__all__ = ["InsuranceEnv"]

# Bucketing constants
AGE_MIN, AGE_MAX = 18, 80
AGE_BIN_SIZE = 5
N_AGE_BINS = (AGE_MAX - AGE_MIN) // AGE_BIN_SIZE + 1
N_REGIONS = 5
RISK_BIN_SIZE = 0.1
N_RISK_BINS = int(1 / RISK_BIN_SIZE)
N_STATE_BUCKETS = N_AGE_BINS * N_REGIONS * N_RISK_BINS

@dataclass(slots=True)
class Profile:
    """Basic customer profile data."""
    age: int       # between 18 and 80
    region: int    # index from 0 to 4
    risk_score: float  # continuous value in [0, 1]

class InsuranceEnv(gym.Env):
    """
    Simple insurance underwriting simulator.
    Actions: reject (0) or accept (1).
    Reward: premium collected minus delayed claim cost (if any).
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        n_profiles: int = 100,
        pareto_alpha: float = 1.5,
        pareto_xm: float = 1.0,
        delay: int = 10,
        horizon: int = 1_000,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()

        # RNG and synthetic profiles
        self.rng = np.random.default_rng(seed)
        self.n_profiles = n_profiles
        self.profiles = [
            Profile(
                age=int(self.rng.integers(AGE_MIN, AGE_MAX + 1)),
                region=int(self.rng.integers(N_REGIONS)),
                risk_score=float(self.rng.random()),
            )
            for _ in range(n_profiles)
        ]

        # Precompute tariffs and claim probabilities
        self.premiums = np.array([self._price(p) for p in self.profiles], dtype=np.float32)
        self.claim_probs = np.array([self._claim_prob(p) for p in self.profiles], dtype=np.float32)

        # Loss distribution and timing
        self.pareto_alpha = pareto_alpha
        self.pareto_xm = pareto_xm
        self.delay = delay
        self.horizon = horizon

        # Define action and observation spaces
        self.action_space = spaces.Discrete(2)  # 0: reject, 1: accept
        self.observation_space = spaces.Discrete(N_STATE_BUCKETS)

        # Buffer holds delayed claim payouts
        self.buffer = deque([0.0] * (delay + 1), maxlen=delay + 1)
        self.current_state = 0
        self.t = 0

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng.bit_generator.seed(seed)
        # Clear any pending payouts
        self.buffer.clear()
        self.buffer.extend([0.0] * (self.delay + 1))
        self.t = 0
        # Sample initial customer
        self.current_state = self._sample_state()
        return np.int64(self.current_state), {}

    def step(self, action: int):
        assert self.action_space.contains(action)

        # Get any payout due now
        reward = self.buffer.popleft()

        # Decide on underwriting action
        idx = self._state_to_profile_idx(self.current_state)
        if action == 1:
            # Accept: collect premium and possibly queue a claim loss
            reward += float(self.premiums[idx])
            if self.rng.random() < self.claim_probs[idx]:
                loss = self._pareto()
                self.buffer.append(-loss)
            else:
                self.buffer.append(0.0)
        else:
            # Reject: no premium, no claim
            self.buffer.append(0.0)

        # Advance time step
        self.t += 1
        terminated = self.t >= self.horizon
        self.current_state = self._sample_state()

        return np.int64(self.current_state), reward, terminated, False, {}

    def _price(self, p: Profile) -> float:
        """Compute premium based on age and region."""
        base = 200.0
        age_comp = 4.0 * (p.age - AGE_MIN)
        region_fees = [0.0, 20.0, 40.0, 60.0, 80.0]
        return base + age_comp + region_fees[p.region]

    def _claim_prob(self, p: Profile) -> float:
        """Estimate claim probability from risk score, age, and region."""
        base = 0.02 + 0.25 * p.risk_score
        age_factor = (p.age - AGE_MIN) / (AGE_MAX - AGE_MIN) * 0.10
        region_risk = [0.00, 0.01, 0.03, 0.05, 0.07]
        prob = base + age_factor + region_risk[p.region]
        return float(np.clip(prob, 0.0, 0.9))

    def _sample_state(self) -> int:
        """Pick a random profile and bucket it."""
        idx = int(self.rng.integers(self.n_profiles))
        return self._encode(self.profiles[idx])

    def _encode(self, p: Profile) -> int:
        """Turn profile into a single bucket ID."""
        age_bin = min((p.age - AGE_MIN) // AGE_BIN_SIZE, N_AGE_BINS - 1)
        risk_bin = min(int(p.risk_score // RISK_BIN_SIZE), N_RISK_BINS - 1)
        return age_bin * (N_REGIONS * N_RISK_BINS) + p.region * N_RISK_BINS + risk_bin

    def _state_to_profile_idx(self, state: int) -> int:
        """Find one profile matching the given bucket."""
        for i, p in enumerate(self.profiles):
            if self._encode(p) == state:
                return i
        return 0

    def _pareto(self) -> float:
        """Sample a heavy-tailed loss from a Pareto distribution."""
        u = self.rng.random()
        return self.pareto_xm / (u ** (1.0 / self.pareto_alpha))
