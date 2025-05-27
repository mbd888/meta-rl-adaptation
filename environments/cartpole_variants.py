import gym
import json
from gym.envs.classic_control.cartpole import CartPoleEnv

class CustomCartPoleEnv(CartPoleEnv):
    def __init__(self, masspole, length):
        super().__init__()
        self.masspole = masspole
        self.length = length
        self.total_mass = self.masspole + self.masscart
        self.polemass_length = self.masspole * self.length

def make_variant(name, config_path="config/env_variants.json"):
    with open(config_path, 'r') as f:
        variants = json.load(f)
    if name not in variants:
        raise ValueError(f"Unknown variant: {name}")
    params = variants[name]
    return CustomCartPoleEnv(**params)