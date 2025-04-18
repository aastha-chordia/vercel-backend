import gym
from gym import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
# Load the CSV file
file_path = "reddit_comments_sorted_by_coin.csv"
df = pd.read_csv(file_path)
# Display the first few rows to inspect columns and data types
print(df.head())
print(df.columns)
print("Actual columns:", df.columns.tolist())

df.columns = df.columns.str.strip().str.lower()

# Show cleaned column names
print("Cleaned columns:", df.columns.tolist())

# Ensure compound_sentiment is numeric
df['compound_sentiment'] = pd.to_numeric(df['compound_sentiment'], errors='coerce')

# Convert timestamp to datetime, coercing errors
df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

# Drop rows with invalid timestamps
df = df.dropna(subset=['timestamp'])

# Set timestamp as index
df.set_index('timestamp', inplace=True)

# Resample compound_sentiment per hour and fill missing values with 0
hourly_sentiment = df['compound_sentiment'].resample('1H').mean().fillna(0)

# Show the first few rows of the result
print(hourly_sentiment.head())

#Simulate Data (Price + Sentiment)

# Simulated price data (timestamp + price with random walk)
price_data = {
    'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='h'),  # use 'h' instead of 'H'
    'price': (10000 + np.random.randn(100).cumsum())  # use np.random
}
price_df = pd.DataFrame(price_data)
price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
price_df.set_index('timestamp', inplace=True)

# Merge the price data with sentiment data
merged_df = price_df.join(hourly_sentiment.rename("sentiment"), how='left').fillna(0)
print(merged_df.head())

# Custom Trading Environment

class CryptoTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super().__init__()
        self.data = data.reset_index()
        self.initial_balance = initial_balance
        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = spaces.Box(
            low=np.array([0, -1, 0, 0], dtype=np.float32),
            high=np.array([np.finfo(np.float32).max]*4, dtype=np.float32),
            dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.current_step = 0
        return self._get_obs()

    def _get_obs(self):
        row = self.data.iloc[self.current_step]
        return np.array([
            row['price'],
            row['sentiment'],
            self.balance,
            self.position
        ], dtype=np.float32)

    def step(self, action):
        row = self.data.iloc[self.current_step]
        price = row['price']
        reward = 0

        # Action logic
        if action == 0:  # Buy
            if self.balance > 0:
                self.position = self.balance / price
                self.balance = 0
        elif action == 1:  # Sell
            if self.position > 0:
                self.balance = self.position * price
                self.position = 0

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        new_price = self.data.iloc[self.current_step]['price']
        new_net_worth = self.balance + self.position * new_price
        old_net_worth = self.balance + self.position * price
        reward = new_net_worth - old_net_worth

        obs = self._get_obs()
        return obs, reward, done, {}

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position:.4f}")


# Deep Q-Network Agent

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


#Training Setup

env = CryptoTradingEnv(merged_df)
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

dqn = DQN(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(dqn.parameters(), lr=1e-3)
replay_buffer = deque(maxlen=2000)

def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    state = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        q_vals = dqn(state)
    return torch.argmax(q_vals).item()


#Training Loop

num_episodes = 300
batch_size = 32
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
reward_history = []

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state

        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(dones).unsqueeze(1)

            q_values = dqn(states).gather(1, actions)
            next_q = dqn(next_states).max(1)[0].unsqueeze(1)
            expected_q = rewards + gamma * next_q * (1 - dones)

            loss = criterion(q_values, expected_q.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    reward_history.append(total_reward)
    print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")


from datetime import datetime  

# to save the rl model
torch.save({
    'model_state_dict': dqn.state_dict(),
    'input_dim': input_dim,
    'output_dim': output_dim,
    'architecture': 'DQN',
    'metadata': {
        'created_at': str(datetime.now()),
        'pytorch_version': torch.__version__
    }
}, 'crypto_trading_dqn_v2.pth')

# Download the new file
from google.colab import files
files.download('crypto_trading_dqn_v2.pth')

#Visualization

plt.plot(reward_history, label='Reward')
plt.plot(pd.Series(reward_history).rolling(10).mean(), label='Moving Avg (10)')
plt.legend()
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Trading Performance")
plt.grid(True)
plt.show()

reward_history.append(total_reward)
plt.plot(reward_history, label='Reward')

class CryptoTradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(CryptoTradingEnv, self).__init__()

        # Data should be a DataFrame with "price" and "sentiment"
        self.data = data.reset_index()  # Reset index to use row-based access
        self.max_steps = len(self.data) - 1
        self.current_step = 0

        # Trading parameters
        self.balance = initial_balance
        self.position = 0  # Number of crypto units held

        # Define action space: 0 = Buy, 1 = Sell, 2 = Hold
        self.action_space = spaces.Discrete(3)

        # Define observation space: [price, sentiment, balance, position]
        # Adjust bounds as needed for your feature scaling
        low = np.array([0, -1, 0, 0], dtype=np.float32)
        high = np.array([np.finfo(np.float32).max, 1, np.finfo(np.float32).max, np.finfo(np.float32).max], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _get_obs(self):
        # Get current price and sentiment from the data
        row = self.data.iloc[self.current_step]
        price = row['price']
        sentiment = row.get('sentiment', 0)
        return np.array([price, sentiment, self.balance, self.position], dtype=np.float32)

    def step(self, action):
        done = False
        current_price = self.data.iloc[self.current_step]['price']

        # Execute action: Buy, Sell, or Hold
        if action == 0:  # Buy: Use all available balance to buy crypto
            if self.balance > 0:
                self.position = self.balance / current_price
                self.balance = 0
        elif action == 1:  # Sell: Sell all crypto holdings
            if self.position > 0:
                self.balance = self.position * current_price
                self.position = 0
        # Hold does nothing

        # Advance one timestep
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        # Calculate reward as net worth change
        net_worth = self.balance + self.position * current_price
        reward = net_worth  # This is a simplistic reward; consider using the profit change instead

        obs = self._get_obs()
        info = {"net_worth": net_worth}

        return obs, reward, done, info

    def reset(self):
        self.current_step = 0
        self.balance = 10000
        self.position = 0
        return self._get_obs()

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position:.4f}")

# Example usage:
env = CryptoTradingEnv(merged_df)
obs = env.reset()
print("Initial observation:", obs)


import gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# Environment
env = gym.make("CartPole-v1")

state = env.reset()

# Define a simple DQN network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Hyperparameters
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
learning_rate = 1e-3
gamma = 0.99
num_episodes = 1000
batch_size = 32
replay_buffer = deque(maxlen=2000)

# Network, loss, optimizer
dqn = DQN(input_dim, output_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)

def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = dqn(state)
        return torch.argmax(q_values).item()

# Training loop
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = select_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

        replay_buffer.append((state, action, reward, next_state, done))
        state = next_state

        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(states)
            actions = torch.LongTensor(actions).unsqueeze(1)
            rewards = torch.FloatTensor(rewards).unsqueeze(1)
            next_states = torch.FloatTensor(next_states)
            dones = torch.FloatTensor(dones).unsqueeze(1)

            q_values = dqn(states).gather(1, actions)
            next_q_values = dqn(next_states).max(1)[0].unsqueeze(1)
            expected_q_values = rewards + gamma * next_q_values * (1 - dones)

            loss = criterion(q_values, expected_q_values.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

# Save model
torch.save(dqn.state_dict(), "dqn_trading_model.pth")

