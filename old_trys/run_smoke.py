from old_trys.insurance_gym import InsuranceEnv
from source.qlearner import QLearner

env = InsuranceEnv(delay=5, seed=0)
agent = QLearner(n_states=env.observation_space.n,
                 n_actions=env.action_space.n,
                 alpha=0.1, gamma=0.99, seed=0)

obs, _ = env.reset()
eps = 0.3
for t in range(20):
    action = agent.act(obs, eps)
    next_obs, reward, *_ = env.step(action)
    agent.update(obs, action, reward, next_obs)
    obs = next_obs
    print(f"t={t:2d} | a={action} | r={reward:+.2f}")
