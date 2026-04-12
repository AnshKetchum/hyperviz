import torch.nn.functional as F 
import torch 
import torch.nn as nn 
import math 
from .trajectory import Trajectory, TrajectoryCollection

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