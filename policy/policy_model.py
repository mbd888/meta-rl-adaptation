import torch
import torch.nn as nn

class MetaConditionedPolicy(nn.Module):
    def __init__(self, obs_dim, context_dim, hidden_dim, action_dim):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(obs_dim + context_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs, context):
        x = torch.cat([obs, context], dim=-1)  # concat state and latent vector
        return self.policy_net(x)
