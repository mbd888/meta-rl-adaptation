
import torch
import torch.nn as nn
import gym
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from encoder.encoder_model import ContextEncoder

class ContextPolicy(nn.Module):
    def __init__(self, obs_dim, context_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs, context):
        x = torch.cat([obs, context.expand(obs.size(0), -1)], dim=-1)
        return self.net(x)

def collect_initial_trajectory(env, steps=15):
    traj = []
    obs, _ = env.reset()
    for _ in range(steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        traj.append(obs[:3])  # Using only 3D input for encoder
        obs = next_obs
        if done:
            obs, _ = env.reset()
    return torch.tensor(traj, dtype=torch.float32).unsqueeze(0)  # (1, time, features)

def train_policy(env, encoder, episodes=500):
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    context_dim = 3  # encoder output
    policy = ContextPolicy(obs_dim, context_dim, 128, action_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    for episode in range(episodes):
        traj = collect_initial_trajectory(env)
        context = encoder(traj).detach()

        obs, _ = env.reset()
        log_probs = []
        rewards = []
        total_reward = 0

        for _ in range(200):
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = policy(obs_tensor, context)
            action_dist = torch.distributions.Categorical(logits=logits)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)

            obs, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            log_probs.append(log_prob)
            rewards.append(reward)
            total_reward += reward
            if done:
                break

        # REINFORCE update
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + 0.99 * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.tensor(discounted_rewards)

        loss = -torch.stack(log_probs) * discounted_rewards
        loss = loss.sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 10 == 0:
            print(f"Episode {episode}, Return: {total_reward:.2f}")
        
        torch.save(policy.state_dict(), "policy/meta_policy.pth")


if __name__ == "__main__":
    encoder = ContextEncoder(input_dim=45, hidden_dim=128, output_dim=3)
    encoder.load_state_dict(torch.load("encoder/context_encoder.pth"))
    encoder.eval()

    env = gym.make("CartPole-v1")
    train_policy(env, encoder)
