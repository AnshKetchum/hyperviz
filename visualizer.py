import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

""" 

- need to have a dataclass for each hidden state "trajectory", each probably will be a [(B, T, E)] * num_layers tensor list
- should consider this across multiple trajectories

how we'll use this: 
1. angles between hidden states
2. angles between subsequent hidden states

"""

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

class Visualizer:
    def __init__(self, save_directory: str):
        self.collection = TrajectoryCollection(trajectories=[])
        self.save_dir = save_directory

    def add(self, trajectory: Trajectory):
        self.collection.add(trajectory)
    
    def load(self, visualization_tensor_path: str):
        assert os.path.exists(visualization_tensor_path)
        self.collection.load(visualization_tensor_path)

    def save(self):
        self.collection.save(os.path.join(self.save_dir, "visualization_tensors.pth"))

    def visualize(self):
        os.makedirs(self.save_dir, exist_ok=True)

        angles  = hidden_state_angle_analysis(self.collection)
        deltas  = hidden_state_delta_analysis(self.collection)
        norms   = hidden_state_relative_norm_analysis(self.collection)
        pca     = hidden_state_pca_analysis(self.collection)

        # 1. angle vs layer
        _, ax = plt.subplots(figsize=(7, 5))
        for a in angles:
            ax.plot(a, color="steelblue", alpha=0.05)
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
            ax.scatter(coords[:, 0], coords[:, 1], color="mediumpurple", s=4, alpha=0.05)
        ax.set_title("hidden state PC1 vs PC2")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "hidden_state_pca.png"), dpi=150)
        plt.close()

        # dump tensors
        self.save()

        
def hidden_state_angle_analysis(trajectories: TrajectoryCollection):
    """
        computes angles via inverse cosine similarity between hidden state i.

        theta = acos(<a, b> / (||a|| * ||b||))

        namely:

        theta(h(i), h(i+1)) =

        then, plots angle vs layer

        and returns a list of vectors T x d, where T is the number of trajectories and d is the dimension of the pairwise angle
        vector.
    """
    all_angles = []
    for traj in trajectories.get():
        hidden_states = traj.get()  # list of (B, T, E)
        angles = []
        for i in range(len(hidden_states) - 1):
            h_i    = hidden_states[i].flatten(0, 1)      # (B*T, E)
            h_next = hidden_states[i + 1].flatten(0, 1)  # (B*T, E)
            cos_sim = F.cosine_similarity(h_i, h_next, dim=-1)
            angle = torch.acos(cos_sim.clamp(-1, 1)).mean().item()
            angles.append(angle)
        all_angles.append(angles)

    return all_angles

def hidden_state_relative_norm_analysis(trajectories: TrajectoryCollection):
    """
        computes the per-layer Frobenius norm of each hidden state relative to the first:

        r(i) = ||h(i)||_F / ||h(0)||_F

        returns a list of vectors T x num_layers, one scalar per layer per trajectory.
    """
    all_norms = []
    for traj in trajectories.get():
        hidden_states = traj.get()  # list of (B, T, E)
        base_norm = torch.norm(hidden_states[0].float(), p='fro').item()
        norms = [torch.norm(h.float(), p='fro').item() / base_norm for h in hidden_states]
        all_norms.append(norms)

    return all_norms

def hidden_state_pca_analysis(trajectories: TrajectoryCollection):
    """
        for each trajectory, averages hidden states over B and T to get one point per layer,
        then projects to PC1 vs PC2 via PCA (SVD).

        returns a list of (num_layers, 2) tensors, one per trajectory.
    """
    all_projections = []
    for traj in trajectories.get():
        hidden_states = traj.get()  # list of (B, T, E)
        # mean over B and T -> (num_layers, E)
        points = torch.stack([h.flatten(0, 1).mean(0) for h in hidden_states]).float()
        points = points - points.mean(0)  # center
        _, _, Vt = torch.linalg.svd(points, full_matrices=False)
        projection = points @ Vt[:2].T  # (num_layers, 2)
        all_projections.append(projection)

    return all_projections

def hidden_state_delta_analysis(trajectories: TrajectoryCollection):
    """
        computes angles via inverse cosine similarity.

        namely:

        theta(h(i) - h(i - 1), h(i+1) - h(i)) =

        then, plots angle vs layer

        and returns a list of vectors T x d, where T is the number of trajectories and d is the dimension of the pairwise delta angle
        vector.
    """
    all_angles = []
    for traj in trajectories.get():
        hidden_states = traj.get()  # list of (B, T, E)
        deltas = [hidden_states[i + 1] - hidden_states[i] for i in range(len(hidden_states) - 1)]
        angles = []
        for i in range(len(deltas) - 1):
            d_i    = deltas[i].flatten(0, 1)      # (B*T, E)
            d_next = deltas[i + 1].flatten(0, 1)  # (B*T, E)
            cos_sim = F.cosine_similarity(d_i, d_next, dim=-1)
            angle = torch.acos(cos_sim.clamp(-1, 1)).mean().item()
            angles.append(angle)
        all_angles.append(angles)

    return all_angles
