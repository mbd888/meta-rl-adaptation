import torch
import numpy as np
from environments.cartpole_variants import make_variant
import os

def collect_trajectory(env, steps=5):
    traj = []
    obs, _ = env.reset()
    for _ in range(steps):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        traj.append(np.concatenate([obs, [action], next_obs]))
        done = terminated or truncated
        obs = env.reset()[0] if done else next_obs
    return np.array(traj)


def generate_data():
    variant_names = ["cartpole_light", "cartpole_medium", "cartpole_heavy"]
    output_dir = "data/trajectories"
    os.makedirs(output_dir, exist_ok=True)

    for name in variant_names:
        env = make_variant(name)
        variant_trajs = []
        for _ in range(100):  # 100 trajectories per variant
            traj = collect_trajectory(env)
            variant_trajs.append(traj)
        variant_trajs = np.stack(variant_trajs)
        out_path = os.path.join(output_dir, f"{name}.pt")
        torch.save(torch.tensor(variant_trajs, dtype=torch.float32), out_path)
        print(f"Saved {name} trajectories to {out_path}")

if __name__ == "__main__":
    generate_data()