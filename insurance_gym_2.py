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
    Reward: premium earned − payouts due now − holding cost on liabilities.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        n_profiles: int = 200,
        horizon: int = 1000,
        seed: Optional[int] = None,
        base_premium: float = 200.0,
        base_price_age: float = 4.0,
        region_fees = (0.0, 20.0, 40.0, 60.0, 80.0),
        pareto_alpha: float = 1.5,
        pareto_xm: float = 1.0,
        delay_min: int = 3,
        delay_max: int = 20,
        capital_init: float = 50_000.0,
        holding_cost: float = 0.00002,   # cost per unit of liability per step
        bankruptcy_penalty: float = 5_000.0,  # extra penalty on bankruptcy
        regime_switch_p: float = 0.05,   # ~expected duration 1/p
        regime_loss_multipliers = (0.8, 1.0, 1.3),
        regime_claim_add = (0.0, 0.02, 0.05),
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.horizon = horizon

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
        self.holding_cost = holding_cost
        self.bankruptcy_penalty = bankruptcy_penalty

        # Regimes
        self.regime_switch_p = regime_switch_p
        self.regime_loss_multipliers = np.array(regime_loss_multipliers, dtype=np.float32)
        self.regime_claim_add = np.array(regime_claim_add, dtype=np.float32)

        # Action / observation spaces
        self.action_space = spaces.Discrete(1 + len(PRICE_FACTORS))  # 0=reject, 1..K=accept@price
        self.observation_space = spaces.Discrete(
            N_PROFILE_BUCKETS * CAP_BINS * LIAB_BINS * N_REGIMES
        )

        # Runtime state
        self.t = 0
        self.current_profile_idx = 0
        self.capital = capital_init
        self.liabilities_total = 0.0
        self.regime = int(self.rng.integers(N_REGIMES))
        # Liability queue: slot[d] holds payouts due in d steps
        self.buffer = deque([0.0] * (self.delay_max + 1), maxlen=self.delay_max + 1)

    # ---------- Helpers ----------
    def _price(self, p: Profile) -> float:
        return float(self.base_premium + self.base_price_age * (p.age - AGE_MIN) + self.region_fees[p.region])

    def _claim_prob(self, p: Profile) -> float:
        base = 0.02 + 0.25 * p.risk_score
        age_factor = (p.age - AGE_MIN) / (AGE_MAX - AGE_MIN) * 0.10
        region_risk = np.array([0.00, 0.01, 0.03, 0.05, 0.07], dtype=np.float32)
        prob = base + age_factor + region_risk[p.region] + self.regime_claim_add[self.regime]
        return float(np.clip(prob, 0.0, 0.95))

    def _pareto(self) -> float:
        u = float(self.rng.random())
        loss = self.pareto_xm / (u ** (1.0 / self.pareto_alpha))
        return loss * float(self.regime_loss_multipliers[self.regime])

    def _encode_profile(self, p: Profile) -> int:
        age_bin = min((p.age - AGE_MIN) // AGE_BIN_SIZE, N_AGE_BINS - 1)
        risk_bin = min(int(p.risk_score // RISK_BIN_SIZE), N_RISK_BINS - 1)
        return int(age_bin * (N_REGIONS * N_RISK_BINS) + p.region * N_RISK_BINS + risk_bin)

    def _bin_capital(self, c: float) -> int:
        # Target scale idea: 0..CAP_BINS-1 on a rough log/quantile scale; here we keep it simple via clipping.
        # Example: [-10k, 0) -> 0, [0, 20k) -> 1, ... >= 100k -> CAP_BINS-1
        edges = np.linspace(-10_000, 100_000, CAP_BINS - 1)
        return int(np.digitize([c], edges)[0])

    def _bin_liabilities(self, L: float) -> int:
        # Simple clipping over [0, 200k] into LIAB_BINS buckets
        edges = np.linspace(0, 200_000, LIAB_BINS - 1)
        return int(np.digitize([L], edges)[0])

    def _encode_obs(self, p: Profile) -> int:
        pb = self._encode_profile(p)
        cb = self._bin_capital(self.capital)
        lb = self._bin_liabilities(self.liabilities_total)
        rid = self.regime
        return ((pb * CAP_BINS + cb) * LIAB_BINS + lb) * N_REGIMES + rid

    # ---------- Gym API ----------
    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self.rng.bit_generator.seed(seed)
        self.t = 0
        self.buffer.clear()
        self.buffer.extend([0.0] * (self.delay_max + 1))
        self.capital = self.capital_init
        self.liabilities_total = 0.0
        self.regime = int(self.rng.integers(N_REGIMES))
        self.current_profile_idx = int(self.rng.integers(self.n_profiles))
        obs = self._encode_obs(self.profiles[self.current_profile_idx])
        return np.int64(obs), {"profile_idx": self.current_profile_idx, "regime": self.regime}

    def step(self, action: int) -> Tuple[np.int64, float, bool, bool, dict]:
        assert self.action_space.contains(action)
        reward = 0.0

        # 1) Payouts due today
        payout = self.buffer.popleft()
        self.buffer.append(0.0)  # important: keep deque length constant
        reward += payout  # payout is negative or 0
        self.capital += payout
        self.liabilities_total = max(0.0, self.liabilities_total + payout)  # payout <= 0

        # 2) Holding cost on outstanding liabilities
        hold_cost = - self.holding_cost * self.liabilities_total
        reward += hold_cost
        self.capital += hold_cost

        # 3) Action on current customer
        p = self.profiles[self.current_profile_idx]
        if action == 0:
            # reject
            pass
        else:
            price_factor = float(PRICE_FACTORS[action - 1])
            premium = self._price(p) * price_factor
            reward += premium
            self.capital += premium
            # Claim with regime-adjusted probability
            if self.rng.random() < self._claim_prob(p):
                loss = - self._pareto()
                delay = int(self.rng.integers(self.delay_min, self.delay_max + 1))
                # Pay in "delay" steps: index 0 = next step -> use delay-1
                # After the append above, len(self.buffer) == self.buffer.maxlen
                idx = min(delay, len(self.buffer))  # guard; should already be <= maxlen
                self.buffer[idx - 1] = self.buffer[idx - 1] + loss
                self.liabilities_total += -loss

        # 4) Time & regime transition
        self.t += 1
        if self.rng.random() < self.regime_switch_p:
            self.regime = int(self.rng.integers(N_REGIMES))

        # 5) Bankruptcy check
        terminated = False
        if self.capital < 0.0:
            reward -= self.bankruptcy_penalty
            terminated = True

        # 6) Episode end
        if self.t >= self.horizon:
            terminated = True

        # 7) Next customer (regime implicitly correlates the distribution via the additive term)
        self.current_profile_idx = int(self.rng.integers(self.n_profiles))
        next_obs = self._encode_obs(self.profiles[self.current_profile_idx])
        info = {"profile_idx": self.current_profile_idx, "regime": self.regime,
                "capital": self.capital, "liabilities": self.liabilities_total}
        return np.int64(next_obs), float(reward), terminated, False, info
