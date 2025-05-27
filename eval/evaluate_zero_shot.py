import torch
import gym
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from encoder.encoder_model import ContextEncoder
from policy.train_policy import ContextPolicy as MetaPolicy, collect_initial_trajectory

# --- Load encoder ---
encoder = ContextEncoder(input_dim=45, hidden_dim=128, output_dim=3)
encoder.load_state_dict(torch.load("encoder/context_encoder.pth"))
encoder.eval()

# --- Load trained policy ---
obs_dim = 4
context_dim = 3
action_dim = 2
policy = MetaPolicy(obs_dim, context_dim, 128, action_dim)
policy.load_state_dict(torch.load("policy/meta_policy.pth"))
policy.eval()

# --- Evaluate on held-out environment ---
env = gym.make("CartPole-v1")

def run_zero_shot_episode(env, encoder, policy):
    traj = collect_initial_trajectory(env, steps=15)
    with torch.no_grad():
        context = encoder(traj)

    obs, _ = env.reset()
    total_reward = 0

    for _ in range(200):  # Max episode length
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = policy(obs_tensor, context)
            action = torch.argmax(logits, dim=-1).item()
        obs, reward, done, *_ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

# Run multiple episodes
returns = []
for ep in range(20):
    ret = run_zero_shot_episode(env, encoder, policy)
    returns.append(ret)
    print(f"Episode {ep}: Return = {ret:.2f}")

print(f"\nAverage Zero-Shot Return on Held-Out Env: {sum(returns) / len(returns):.2f}")
