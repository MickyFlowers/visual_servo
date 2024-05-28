import torch
import torch.nn as nn

class control_policy(nn.Module):
    def __init__(self, input_dim):
        super(control_policy, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.vel_dir = nn.Linear(128, 6)
        self.log_vel_norm = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.policy(x)
        return self.vel_dir(x), self.log_vel_norm(x) 
    
class control_policy_v2(nn.Module):
    def __init__(self, input_dim):
        super(control_policy_v2, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.vel_dir = nn.Linear(128, 6)
        self.vel_norm = nn.Linear(128, 1)
    
    def forward(self, x):
        x = self.policy(x)
        return torch.tanh(self.vel_dir(x)), torch.sigmoid(self.vel_norm(x))