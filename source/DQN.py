import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os

"""
Implementation of a DQN agent with experience replay and target network.
"""

class ReplayBuffer:
    """
    Ring buffer for storing experience tuples
    """
    def __init__(self, capacity: int, seed: int | None = None):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def push(self, state, action, reward, next_state, done):
        """
        Add new transition, overwrite oldest if buffer full.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        """
        Sample batch of transitions and convert to tensors efficiently.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states_arr = np.array(states, dtype=np.float32)       # shape (batch, state_dim)
        actions_arr = np.array(actions, dtype=np.int64)       # shape (batch,)
        rewards_arr = np.array(rewards, dtype=np.float32)     # shape (batch,)
        next_states_arr = np.array(next_states, dtype=np.float32)  # shape (batch, state_dim)
        dones_arr = np.array(dones, dtype=np.float32)         # shape (batch,)
        return (
            torch.from_numpy(states_arr),                  # Tensor states
            torch.from_numpy(actions_arr),                 # Tensor actions
            torch.from_numpy(rewards_arr),                 # Tensor rewards
            torch.from_numpy(next_states_arr),             # Tensor next_states
            torch.from_numpy(dones_arr),                   # Tensor dones
        )

    def __len__(self):
        return len(self.buffer)  # current number of stored transitions

class QNetwork(nn.Module):
    """
    Feedforward network mapping state -> Q-values.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dims=(128, 128), seed: int | None = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))  # hidden layer
            layers.append(nn.ReLU())           # activation
            prev = h
        layers.append(nn.Linear(prev, output_dim))  # output layer
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # return Q-values for each action

class DQNAgent:
    """
    Wrapper for policy & target networks, action selection, and training.
    """
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_dims=(128,128),
        buffer_capacity=10000,
        batch_size=64,
        gamma=0.99,
        lr=1e-3,
        sync_every=1000,
        device=None,
        seed: int | None = None # random seed for reproducibility
    ):
        
        # Set all seeds for reproducibility

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        # Setup device and networks
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = QNetwork(state_size, action_size, hidden_dims, seed=seed).to(self.device)
        self.target_net = QNetwork(state_size, action_size, hidden_dims, seed=seed).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())  # copy weights
        self.target_net.eval()  # set target net to evaluation mode

        # Optimizer and replay buffer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity, seed=seed)
        self.batch_size = batch_size
        self.gamma = gamma
        self.sync_every = sync_every
        self.step_count = 0

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        """
        Epsilon-greedy: random action or best Q-value.
        """
        exploration = False
        if random.random() < epsilon:
            exploration = True
            return random.randrange(self.policy_net.net[-1].out_features), exploration
        state_v = torch.from_numpy(state).unsqueeze(0).to(self.device)  # add batch dim
        q_vals = self.policy_net(state_v)  # compute Q-values
        return int(q_vals.argmax().item()), exploration  # select argmax

    def store_transition(self, state, action, reward, next_state, done):
        # Add transition to replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)

    def optimize(self):
        """
        Only train if enough samples.
        """
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.unsqueeze(-1).to(self.device)  # shape (batch,1)
        rewards = rewards.unsqueeze(-1).to(self.device)  # shape (batch,1)
        next_states = next_states.to(self.device)
        dones = dones.unsqueeze(-1).to(self.device)      # shape (batch,1)

        # Compute current Q(s,a)
        q_values = self.policy_net(states).gather(1, actions)

        # Compute target Q using target network
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
        target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss and backprop
        loss = nn.functional.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Periodic target network update
        self.step_count += 1
        if self.step_count % self.sync_every == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())  # sync networks

    def save_model(self, save_dir: str, filename: str):
        """
        Save the policy network weights to disk.
        """
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(base_dir, "results")
        save_path = os.path.join(results_dir, save_dir)
        save_path = os.path.join(save_dir, "models")

        os.makedirs(save_path, exist_ok=True)
        full_path = os.path.join(save_path, filename)

        torch.save(self.policy_net.state_dict(), full_path)
        print(f"[DQNAgent] Model saved to {full_path}")