import torch
import gym
import json
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from encoder.encoder_model import ContextEncoder
from policy.train_policy import ContextPolicy as MetaPolicy, collect_initial_trajectory
from environments.cartpole_variants import make_variant

encoder = ContextEncoder(input_dim=45, hidden_dim=128, output_dim=3)
encoder.load_state_dict(torch.load("encoder/context_encoder.pth"))
encoder.eval()

obs_dim = 4
context_dim = 3
action_dim = 2
policy = MetaPolicy(obs_dim, context_dim, 128, action_dim)
policy.load_state_dict(torch.load("policy/meta_policy.pth"))
policy.eval()

def run_zero_shot_episode(env, encoder, policy):
    traj = collect_initial_trajectory(env, steps=15)
    with torch.no_grad():
        context = encoder(traj)

    obs, _ = env.reset()
    total_reward = 0
    for _ in range(200):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = policy(obs_tensor, context)
            action = torch.argmax(logits, dim=-1).item()
        obs, reward, done, *_ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

variant_names = ["cartpole_light", "cartpole_medium", "cartpole_heavy"]
all_returns = {}

for variant in variant_names:
    print(f"\n--- Evaluating {variant} ---")
    env = make_variant(variant)
    returns = [run_zero_shot_episode(env, encoder, policy) for _ in range(20)]
    avg = sum(returns) / len(returns)
    print("\n".join([f"Episode {i}: Return = {r:.2f}" for i, r in enumerate(returns)]))
    print(f"Average Return on {variant}: {avg:.2f}")
    all_returns[variant] = returns

print("\n--- Evaluating Unseen Variant ---")
env = gym.make("CartPole-v1")
returns = [run_zero_shot_episode(env, encoder, policy) for _ in range(20)]
avg = sum(returns) / len(returns)
print("\n".join([f"Episode {i}: Return = {r:.2f}" for i, r in enumerate(returns)]))
print(f"Average Return on Unseen: {avg:.2f}")
all_returns["unseen"] = returns

os.makedirs("eval", exist_ok=True)
with open("eval/returns.json", "w") as f:
    json.dump(all_returns, f, indent=2)
