import torch
import torch.nn as nn

class ContextEncoder(nn.Module):
    def __init__(self, input_dim=45, hidden_dim=128, output_dim=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, traj_batch):
        # traj_batch shape: (batch_size, time_steps, input_dim)
        batch_size, time_steps, input_dim = traj_batch.size()
        traj_flat = traj_batch.view(batch_size, -1)  # Flatten time
        return self.encoder(traj_flat)