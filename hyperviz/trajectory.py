import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

# simple dataclass
class Trajectory: 
    def __init__(self, hidden_states: list[torch.Tensor] = []): 
        self.hidden_states = []

        # safely add them
        for t in hidden_states:
            self.add(t)


    def add(self, tensor: torch.Tensor):
        assert all([tensor.shape == t.shape for t in self.hidden_states])
        self.hidden_states.append(tensor)
    
    def get(self):
        return self.hidden_states

# efficiently store a bunch of hidden state trajectories
class TrajectoryCollection:
    def __init__(self, trajectories: list[Trajectory] = []):
        self.trajectories = trajectories

    def add(self, trajectory: Trajectory):
        self.trajectories.append(trajectory)
    
    def get(self):
        return self.trajectories
    
    def save(self, filepath: str):
        data = {"trajectories": [traj.get() for traj in self.trajectories]}
        torch.save(data, filepath)

    @classmethod
    def load(cls, filepath: str) -> "TrajectoryCollection":
        data = torch.load(filepath)
        trajectories = [Trajectory(hidden_states=tensors) for tensors in data["trajectories"]]
        return cls(trajectories=trajectories)