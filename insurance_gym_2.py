import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

# Customer bucketing (same approach as before)
AGE_MIN, AGE_MAX = 18, 80
AGE_BIN_SIZE = 5
N_AGE_BINS = (AGE_MAX - AGE_MIN) // AGE_BIN_SIZE + 1
N_REGIONS = 5
RISK_BIN_SIZE = 0.1
N_RISK_BINS = int(1 / RISK_BIN_SIZE)
N_PROFILE_BUCKETS = N_AGE_BINS * N_REGIONS * N_RISK_BINS

# Additional buckets
CAP_BINS = 10
LIAB_BINS = 10
N_REGIMES = 3  # e.g., good / neutral / bad

PRICE_FACTORS = np.array([0.9, 1.0, 1.1, 1.2], dtype=np.float32)  # discrete price levels

@dataclass(slots=True)
class Profile:
    age: int
    region: int
    risk_score: float

class InsuranceEnvV2(gym.Env):
    """
    RL environment for insurance pricing/underwriting.

    - Tracks capital (can go bankrupt)
    - Risk switches between a few discrete regimes
    - Action: reject or accept at one of several price factors
    - Claims arrive with delays; unpaid claims sit as liabilities

    Observation: single discrete index encoding (profile bucket, capital bin, liability bin, regime).
    Action: 0 = reject; 1..K = accept at PRICE_FACTORS[i-1].
    Reward: premium earned - payouts due now
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        n_profiles: int = 3000, # number of unique customer profiles
        horizon: int = 1000, # maximum number of steps per episode
        seed: Optional[int] = None, # random seed for reproducibility
        base_premium: float = 200.0, # base premium for all customers
        base_price_age: float = 4.0, # price per year of age 
        region_fees = (0.0, 20.0, 40.0, 60.0, 80.0), # region-specific fees
        pareto_alpha: float = 1.35, # shape parameter for Pareto distribution
        pareto_xm: float = 1.1, # scale parameter for Pareto distribution
        delay_min: int = 5, # min delay for claim payouts
        delay_max: int = 25, # max delay for claim payouts
        capital_init: float = 50_000.0, # initial capital
        bankruptcy_penalty: float = 100_000.0,  # extra penalty on bankruptcy
        regime_switch_p: float = 0.05,   # ~expected duration 1/p
        regime_loss_multipliers = (0.75, 1.0, 1.25), # multipliers for Pareto losses in each regime
        regime_claim_add = (0.00, 0.10, 0.20),   # additive claim probability per regime
        terminal_settle: bool = True, # whether to settle all liabilities at episode end
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.horizon = horizon
        self.terminal_settle = bool(terminal_settle)

        # Customer population
        self.n_profiles = n_profiles
        self.profiles = [
            Profile(
                age=int(self.rng.integers(AGE_MIN, AGE_MAX + 1)),
                region=int(self.rng.integers(N_REGIONS)),
                risk_score=float(self.rng.random())
            )
            for _ in range(n_profiles)
        ]
        self.base_premium = base_premium
        self.base_price_age = base_price_age
        self.region_fees = np.array(region_fees, dtype=np.float32)

        # Loss distribution
        self.pareto_alpha = pareto_alpha
        self.pareto_xm = pareto_xm
        self.delay_min = delay_min
        self.delay_max = delay_max

        # Capital & liabilities
        self.capital_init = capital_init
        self.bankruptcy_penalty = bankruptcy_penalty

        # Regimes
        self.regime_switch_p = regime_switch_p
        self.regime_loss_multipliers = np.array(regime_loss_multipliers, dtype=np.float32)
        self.regime_claim_add = np.array(regime_claim_add, dtype=np.float32)

        # Action / observation spaces
        self.action_space = spaces.Discrete(1 + len(PRICE_FACTORS))  # 0=reject, 1..K=accept@price
        low = np.concatenate([
            np.array([AGE_MIN], dtype=np.float32),
            np.zeros(N_REGIONS, dtype=np.float32),
            np.array([0.0, -np.inf, 0.0, 0.0], dtype=np.float32),
        ])
        high = np.concatenate([
            np.array([AGE_MAX], dtype=np.float32),
            np.ones(N_REGIONS, dtype=np.float32),
            np.array([1.0, np.inf, np.inf, float(N_REGIMES - 1)], dtype=np.float32),
        ])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Runtime state
        self.t = 0
        self.current_profile_idx = 0
        self.capital = capital_init
        self.liabilities_total = 0.0
        self.regime = int(self.rng.integers(N_REGIMES))
        # Liability queue: slot[d] holds payouts due in d steps
        self.buffer = deque([0.0] * (self.delay_max + 1), maxlen=self.delay_max + 1)

    # ---------- Helpers ----------

    
    def _terminal_settlement(self) -> float:
        # Settle all remaining liabilities at episode end
        remaining = float(np.sum(self.buffer))
        if remaining != 0.0:
            self.capital += remaining
            self.liabilities_total = max(0.0, self.liabilities_total + remaining)
            self.buffer.clear()
            self.buffer.extend([0.0] * (self.delay_max + 1))
        return remaining

    def _price(self, p: Profile) -> float:
        # Calculate the premium based on age, region, and risk score
        return float(self.base_premium + self.base_price_age * (p.age - AGE_MIN) + self.region_fees[p.region])

    def _claim_prob(self, p: Profile) -> float:
        # Calculate the claim probability based on profile and regime
        base = 0.02 + 0.1 * p.risk_score # base probability + risk score influence
        age_factor = (p.age - AGE_MIN) / (AGE_MAX - AGE_MIN) * 0.2
        region_risk = np.array([0.00, -0.1, 0.8, -0.2, 0.3], dtype=np.float32)
        prob = base + age_factor + region_risk[p.region] + self.regime_claim_add[self.regime] 
        return float(np.clip(prob, 0.0, 0.80)) 

    def _pareto(self) -> float:
        # Sample a loss from the Pareto distribution
        u = float(self.rng.random())
        loss = self.pareto_xm / (u ** (1.0 / self.pareto_alpha))# sample from Pareto (xm, alpha)
        #print(f"Sampled loss: {loss} with regime multiplier {self.regime_loss_multipliers[self.regime]} is {loss * float(self.regime_loss_multipliers[self.regime])}")
        return loss * float(self.regime_loss_multipliers[self.regime]) * float(self.base_premium)

    def _one_hot_region(self, r: int) -> np.ndarray:
        # One-hot encode the region index
        v = np.zeros(N_REGIONS, dtype=np.float32)
        v[int(r)] = 1.0
        return v

    def _get_obs(self, p: Profile) -> np.ndarray:
        # Get the observation vector for a profile
        reg_oh = self._one_hot_region(p.region)
        return np.concatenate((
            np.array([float(p.age)], dtype=np.float32),
            reg_oh.astype(np.float32),
            np.array([
                float(p.risk_score),
                float(self.capital),
                float(self.liabilities_total),
                float(self.regime),
            ], dtype=np.float32),
        )).astype(np.float32)

    # ---------- Gym API ----------
    def reset(self, *, seed: Optional[int] = None, options=None):
        # Reset the environment state
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        self.buffer.clear()
        self.buffer.extend([0.0] * (self.delay_max + 1))
        self.capital = self.capital_init
        self.liabilities_total = 0.0
        self.regime = int(self.rng.integers(N_REGIMES))
        self.current_profile_idx = int(self.rng.integers(self.n_profiles))
        obs = self._get_obs(self.profiles[self.current_profile_idx])
        return obs, {"profile_idx": self.current_profile_idx, "regime": self.regime}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Execute one step in the environment
        assert self.action_space.contains(action) # check if action is valid
        reward = 0.0

        # Payouts due today
        payout = self.buffer.popleft()
        self.buffer.append(0.0)  # important: keep deque length constant
        reward += payout  # payout is negative or 0
        self.capital += payout
        self.liabilities_total = max(0.0, self.liabilities_total + payout)  # payout <= 0

        # Action on current customer
        p = self.profiles[self.current_profile_idx] # get current profile
        if action == 0:
            # reject
            pass
        else:
            price_factor = float(PRICE_FACTORS[action - 1]) # get price factor for action
            premium = self._price(p) * price_factor
            reward += premium
            self.capital += premium
            # Claim with regime-adjusted probability
            if self.rng.random() < self._claim_prob(p):
                loss = - self._pareto()
                delay = int(self.rng.integers(self.delay_min, self.delay_max + 1))
                # Pay in "delay" steps: index 0 = next step -> use delay-1
                # After the append above, len(self.buffer) == self.buffer.maxlen
                idx = min(delay, len(self.buffer))  # ensure we don't overflow
                self.buffer[idx - 1] = self.buffer[idx - 1] + loss # accumulate losses
                self.liabilities_total += -loss

        # Time & regime transition
        self.t += 1 # increment time step
        if self.rng.random() < self.regime_switch_p:
            self.regime = int(self.rng.integers(N_REGIMES))

        # Bankruptcy check
        terminated = False
        if self.capital < 0.0:
            reward -= self.bankruptcy_penalty
            terminated = True

        # Episode end
        terminal_paid = 0.0
        if self.t >= self.horizon:
            if self.terminal_settle:
                terminal_paid = self._terminal_settlement()
                reward += terminal_paid
                # Check bankruptcy after settlement
                if not terminated and self.capital < 0.0:
                    reward -= self.bankruptcy_penalty
                    terminated = True
            terminated = True

        # Next customer (regime implicitly correlates the distribution via the additive term)
        self.current_profile_idx = int(self.rng.integers(self.n_profiles)) # sample next profile
        next_obs = self._get_obs(self.profiles[self.current_profile_idx]) # get next observation
        info = {"profile_idx": self.current_profile_idx, "regime": self.regime,
                "capital": self.capital, "liabilities": self.liabilities_total, "terminal_paid": float(terminal_paid)}
        return next_obs, float(reward), terminated, False, info
