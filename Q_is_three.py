
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gym

# Define the neural network architecture for the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the Triple DQN agent
class TripleDQNAgent:
    def __init__(self, state_size, action_size, hidden_size=64, gamma=0.99, tau=0.001, lr=0.001, batch_size=64, buffer_size=int(1e5)):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.memory = deque(maxlen=buffer_size)
        self.loss_fn = nn.MSELoss()
        
        # Initialize three Q-networks and target networks
        self.Q1 = QNetwork(state_size, action_size, hidden_size)
        self.Q1_target = QNetwork(state_size, action_size, hidden_size)
        self.Q2 = QNetwork(state_size, action_size, hidden_size)
        self.Q2_target = QNetwork(state_size, action_size, hidden_size)
        self.Q3 = QNetwork(state_size, action_size, hidden_size)
        self.Q3_target = QNetwork(state_size, action_size, hidden_size)
        
        # Copy parameters from Q-networks to target networks
        self.update_target_networks(1.0)
        
        # Initialize optimizer for each Q-network
        self.optimizer1 = optim.Adam(self.Q1.parameters(), lr=lr)
        self.optimizer2 = optim.Adam(self.Q2.parameters(), lr=lr)
        self.optimizer3 = optim.Adam(self.Q3.parameters(), lr=lr)

    def update_target_networks(self, tau):
        for target_param, param in zip(self.Q1_target.parameters(), self.Q1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.Q2_target.parameters(), self.Q2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.Q3_target.parameters(), self.Q3.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def act(self, state, epsilon=0.0):
        if random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                Q_avg = (self.Q1(state) + self.Q2(state) + self.Q3(state)) / 3
                action = Q_avg.argmax(1).item()
        else:
            action = random.choice(np.arange(self.action_size))
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample a mini-batch from the replay buffer
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Compute target Q-values using the minimum of target Q-values
        with torch.no_grad():
            Q1_targets_next = self.Q1_target(next_states)
            Q2_targets_next = self.Q2_target(next_states)
            Q3_targets_next = self.Q3_target(next_states)
            min_Q_targets_next = torch.min(Q1_targets_next, torch.min(Q2_targets_next, Q3_targets_next))
            target_values = rewards + (1 - dones) * self.gamma * min_Q_targets_next

        # Compute Q-values
        Q1_values = self.Q1(states).gather(1, actions)
        Q2_values = self.Q2(states).gather(1, actions)
        Q3_values = self.Q3(states).gather(1, actions)

        # Compute loss for each Q-network
        loss1 = self.loss_fn(Q1_values, target_values)
        loss2 = self.loss_fn(Q2_values, target_values)
        loss3 = self.loss_fn(Q3_values, target_values)

        # Update Q1, Q2, or Q3 based on a random selection among them
        random_choice = random.choice([1, 2, 3])
        if random_choice == 1:
            self.optimizer1.zero_grad()
            loss1.backward()
            self.optimizer1.step()
        elif random_choice == 2:
            self.optimizer2.zero_grad()
            loss2.backward()
            self.optimizer2.step()
        else:
            self.optimizer3.zero_grad()
            loss3.backward()
            self.optimizer3.step()

        # Update target networks
        self.update_target_networks(self.tau)

# Define a function to train the agent on the CartPole environment
def train_agent(env, agent, episodes=5000, max_steps=200, epsilon_start=1.0, epsilon_decay=0.995, epsilon_min=0.01):
    scores = []
    epsilon = epsilon_start
    for episode in range(episodes):
        state = env.reset()
        score = 0
        for step in range(max_steps):
            action = agent.act(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.learn()
            score += reward
            state = next_state
            if done:
                break
        scores.append(score)
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        print(f"Episode {episode + 1}/{episodes}, Score: {score}, Epsilon: {epsilon:.4f}")
        if np.mean(scores[-100:]) >= 195:
            print(f"Environment solved in {episode + 1} episodes!")
            break
    return scores

# Main function
if __name__ == "__main__":
    # Initialize the environment
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initialize the agent
    agent = TripleDQNAgent(state_size, action_size)

    # Train the agent
    scores = train_agent(env, agent)

    # Visualize the agent's performance
    state = env.reset()
    done = False
    while not done:
        action = agent.act(state)
        next_state, _, done, _ = env.step(action)
        env.render()
        state = next_state
        env.render()
        time.sleep(0.02)
    env.close()
