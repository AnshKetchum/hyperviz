""" 
Implementation of https://arxiv.org/pdf/1712.09913

Idea: sample random directional vectors that match the same of each NN's param, and then norm match in the following ways:
    - for convs, norm match each individual filter
    - for fcs, norm match each row, which is effectively a "filter"

"""
import os
import time
import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def normalize_filter(d_filter: torch.Tensor, w_filter: torch.Tensor):
    """Scale random filter d so its Frobenius norm matches w's norm."""
    # d_ij  ←  d_ij * (||w_ij|| / ||d_ij||)
    d_norm = d_filter.norm() + 1e-10
    w_norm = w_filter.norm()
    d_filter.mul_(w_norm / d_norm)

def make_random_direction(params):
    direction = [torch.randn_like(p) for p in params]

    for d, p in zip(direction, params):
        if d.dim() <= 1:
            # Bias / BN scalars — zero them out (ignore, as in the paper)
            d.zero_()
        elif d.dim() == 2:
            # FC layer: each row = one neuron = one "filter"
            for d_row, p_row in zip(d, p):
                normalize_filter(d_row, p_row)
        else:
            # Conv layer: d[out_ch] is the jth filter of this layer
            for d_filt, p_filt in zip(d, p):
                normalize_filter(d_filt, p_filt)

    return direction

def perturb_and_eval(model, original_params, dx, dy, alpha, beta,
                     criterion, loader, device, eval_batches):
    """
    Set model weights to  θ* + α·dx + β·dy  and return mean loss
    over EVAL_BATCHES mini-batches.
    """
    # Overwrite model parameters
    with torch.no_grad():
        for p, w, d0, d1 in zip(model.parameters(), original_params, dx, dy):
            p.copy_(w.to(device) + alpha * d0.to(device) + beta * d1.to(device))

    model.eval()
    total_loss, count = 0.0, 0
    data_iter = iter(loader)
    with torch.no_grad():
        for _ in range(eval_batches):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            loss = criterion(model(inputs), targets)
            total_loss += loss.item()
            count += 1

    return total_loss / max(count, 1)

def compute_landscape(model, original_params, loader, device,
                      criterion=None, grid_range=1.0, grid_points=20,
                      eval_batches=50):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    # Two independent filter-normalized random directions
    print("\nGenerating filter-normalized random directions ...")
    params_cpu = [p.detach().cpu() for p in original_params]
    dx = make_random_direction(params_cpu)
    dy = make_random_direction(params_cpu)

    alphas = np.linspace(-grid_range, grid_range, grid_points)
    betas  = np.linspace(-grid_range, grid_range, grid_points)
    loss_grid = np.zeros((grid_points, grid_points))

    total_evals = grid_points * grid_points
    t0 = time.time()

    print(f"Sweeping {grid_points}×{grid_points} grid "
          f"({total_evals} evaluations) ...")

    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            loss_grid[i, j] = perturb_and_eval(
                model, original_params, dx, dy,
                alpha, beta, criterion, loader, device, eval_batches
            )

        elapsed = time.time() - t0
        frac = (i + 1) / grid_points
        eta = elapsed / frac - elapsed
        print(f"  row {i+1:2d}/{grid_points}  "
              f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s")

    # Restore original weights
    with torch.no_grad():
        for p, w in zip(model.parameters(), original_params):
            p.copy_(w)

    return alphas, betas, loss_grid


# ──────────────────────────────────────────────
# 7. Plot
# ──────────────────────────────────────────────
def plot_landscape(alphas, betas, loss_grid, train_losses=None,
                   grid_points=None, grid_range=None, out_dir="."):
    A, B = np.meshgrid(alphas, betas, indexing='ij')
    Z = loss_grid
    if grid_points is None:
        grid_points = len(alphas)
    if grid_range is None:
        grid_range = float(np.abs(alphas).max())

    fig = plt.figure(figsize=(18, 7))
    fig.patch.set_facecolor('#0d0d0d')

    # ── 3D surface ──────────────────────────────
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_facecolor('#0d0d0d')

    surf = ax1.plot_surface(
        A, B, Z,
        cmap='inferno',
        edgecolor='none',
        alpha=0.92,
        rstride=1, cstride=1,
    )

    ax1.set_title('Loss Landscape (3D Surface)',
                  color='white', fontsize=13, pad=12)
    ax1.set_xlabel('α  (direction 1)', color='#aaaaaa', labelpad=8)
    ax1.set_ylabel('β  (direction 2)', color='#aaaaaa', labelpad=8)
    ax1.set_zlabel('Loss',             color='#aaaaaa', labelpad=8)
    ax1.tick_params(colors='#888888', labelsize=7)
    ax1.xaxis.pane.fill = False
    ax1.yaxis.pane.fill = False
    ax1.zaxis.pane.fill = False
    for pane in (ax1.xaxis.pane, ax1.yaxis.pane, ax1.zaxis.pane):
        pane.set_edgecolor('#333333')

    cbar = fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=12, pad=0.1)
    cbar.ax.yaxis.set_tick_params(color='#aaaaaa')
    cbar.outline.set_edgecolor('#444444')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#aaaaaa', fontsize=7)

    # ── 2D contour ──────────────────────────────
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_facecolor('#111111')

    levels = np.linspace(Z.min(), Z.max(), 25)
    cf = ax2.contourf(A, B, Z, levels=levels, cmap='inferno')
    ax2.contour(A, B, Z, levels=levels, colors='white', linewidths=0.3, alpha=0.4)

    # Mark the θ* (centre of the plot)
    ax2.scatter([0], [0], c='cyan', s=80, zorder=5, label='θ* (trained weights)')
    ax2.legend(facecolor='#1a1a1a', edgecolor='#444444',
               labelcolor='white', fontsize=8)

    ax2.set_title('Loss Landscape (Contour)', color='white', fontsize=13)
    ax2.set_xlabel('α  (direction 1)', color='#aaaaaa')
    ax2.set_ylabel('β  (direction 2)', color='#aaaaaa')
    ax2.tick_params(colors='#888888')
    for spine in ax2.spines.values():
        spine.set_edgecolor('#333333')

    cbar2 = fig.colorbar(cf, ax=ax2, shrink=0.85, aspect=20)
    cbar2.ax.yaxis.set_tick_params(color='#aaaaaa')
    cbar2.outline.set_edgecolor('#444444')
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color='#aaaaaa', fontsize=7)

    plt.suptitle(
        f'Loss Landscape — filter-normalized random directions\n'
        f'{grid_points}×{grid_points} grid  |  range ±{grid_range}',
        color='white', fontsize=11, y=1.01
    )

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "loss_landscape.png")
    plt.savefig(out, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"\nPlot saved → {out}")
    plt.close()

    # ── Training curve (optional) ────────────────
    if train_losses is not None:
        fig2, ax = plt.subplots(figsize=(8, 3), facecolor='#0d0d0d')
        ax.set_facecolor('#0d0d0d')
        ax.plot(train_losses, color='#ff6b35', linewidth=0.8)
        ax.set_title('Training Loss Curve', color='white')
        ax.set_xlabel('Iteration', color='#aaaaaa')
        ax.set_ylabel('Loss',      color='#aaaaaa')
        ax.tick_params(colors='#888888')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
        plt.tight_layout()
        curve_out = os.path.join(out_dir, "training_curve.png")
        plt.savefig(curve_out, dpi=150, bbox_inches='tight',
                    facecolor=fig2.get_facecolor())
        print(f"Training curve saved → {curve_out}")
        plt.close()