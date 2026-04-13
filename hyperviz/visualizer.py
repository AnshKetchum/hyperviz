import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

from .trajectory import Trajectory, TrajectoryCollection
from .analysis import hidden_state_angle_analysis, hidden_state_delta_analysis, hidden_state_pca_analysis, hidden_state_relative_norm_analysis

""" 

- need to have a dataclass for each hidden state "trajectory", each probably will be a [(B, T, E)] * num_layers tensor list
- should consider this across multiple trajectories

how we'll use this: 
1. angles between hidden states
2. angles between subsequent hidden states

"""

class Visualizer:
    def __init__(self, save_directory: str):
        self.collection = TrajectoryCollection(trajectories=[])
        self.save_dir = save_directory

    def add(self, trajectory: Trajectory):
        self.collection.add(trajectory)
    
    def load(self, visualization_tensor_path: str):
        assert os.path.exists(visualization_tensor_path)
        self.collection.load(visualization_tensor_path)
    
    def clear(self):
        self.collection = TrajectoryCollection(trajectories=[])

    def save(self):
        self.collection.save(os.path.join(self.save_dir, "visualization_tensors.pth"))

    def visualize(self):
        print("Visualizing", len(self.collection), "trajectories with lens", [len(s) for s in self.collection.get()])
        os.makedirs(self.save_dir, exist_ok=True)

        angles  = hidden_state_angle_analysis(self.collection)
        deltas  = hidden_state_delta_analysis(self.collection)
        norms   = hidden_state_relative_norm_analysis(self.collection)
        pca     = hidden_state_pca_analysis(self.collection)

        # 1. angle vs layer
        _, ax = plt.subplots(figsize=(7, 5))
        for a in angles:
            ax.plot(a, color="steelblue", alpha=0.05)
            ax.plot(a, color="steelblue", marker="o", markersize=3, linestyle="none", alpha=0.15, zorder=3)
        ax.set_title("hidden state angle vs layer")
        ax.set_xlabel("step")
        ax.set_ylabel("angle (rad)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "hidden_state_angle.png"), dpi=150)
        plt.close()

        # 2. delta angle vs layer
        _, ax = plt.subplots(figsize=(7, 5))
        for d in deltas:
            ax.plot(d, color="darkorange", alpha=0.05)
            ax.plot(d, color="darkorange", marker="o", markersize=3, linestyle="none", alpha=0.15, zorder=3)
        ax.set_title("hidden state difference angle vs layer")
        ax.set_xlabel("step")
        ax.set_ylabel("angle (rad)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "hidden_state_delta_angle.png"), dpi=150)
        plt.close()

        # 3. relative norm vs layer
        _, ax = plt.subplots(figsize=(7, 5))
        for n in norms:
            ax.plot(n, color="seagreen", alpha=0.05)
            ax.plot(n, color="seagreen", marker="o", markersize=3, linestyle="none", alpha=0.15, zorder=3)
        ax.set_title("hidden state relative norm vs layer")
        ax.set_xlabel("step")
        ax.set_ylabel("||h_i||_F / ||h_0||_F")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "hidden_state_rel_norm.png"), dpi=150)
        plt.close()

        # 4. PC1 vs PC2
        _, ax = plt.subplots(figsize=(7, 5))
        for proj in pca:
            coords = proj.numpy()
            ax.scatter(coords[:, 0], coords[:, 1], color="mediumpurple", s=4, alpha=0.15, zorder=3)
        ax.set_title("hidden state PC1 vs PC2")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "hidden_state_pca.png"), dpi=150)
        plt.close()

        # dump tensors
        self.save()

        