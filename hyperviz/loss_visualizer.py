import time
import os

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .loss_landscape import make_random_direction



class LossVisualizer:
    """
    Visualizes the loss landscape of a model by perturbing its parameters
    along two filter-normalized random directions (Li et al., 2018).

    All hyperparameters are fixed at init time; call visualize() to run.
    """

    def __init__(
        self,
        save_directory: str,
        criterion: nn.Module = None,
        grid_points: int = 20,
        grid_range: float = 1.0,
        eval_batches: int = 50,
        save_interactive_visualization: bool = False,
    ):
        """
        Args:
            save_directory:               Where plots and tensors are saved.
            criterion:                    Loss function. Defaults to CrossEntropyLoss.
            grid_points:                  Number of grid points along each axis (N×N grid).
            grid_range:                   α and β are swept over [-grid_range, +grid_range].
            eval_batches:                 Mini-batches used to estimate loss at each grid point.
            save_interactive_visualization: When True, saves an interactive 3D HTML file
                                          (loss_landscape_3d.html) that can be opened in
                                          any browser and rotated/zoomed freely. Requires plotly.
        """
        self.save_dir = save_directory
        self.criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
        self.grid_points = grid_points
        self.grid_range = grid_range
        self.eval_batches = eval_batches
        self.save_interactive_visualization = save_interactive_visualization

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _eval_loss(self, model, loader, device) -> float:
        """Evaluate mean loss over up to eval_batches mini-batches."""
        model.eval()
        total, count = 0.0, 0
        data_iter = iter(loader)
        with torch.no_grad():
            for _ in range(self.eval_batches):
                try:
                    inputs, targets = next(data_iter)
                except StopIteration:
                    break
                inputs, targets = inputs.to(device), targets.to(device)
                total += self.criterion(model(inputs), targets).item()
                count += 1
        return total / max(count, 1)

    def _perturb_and_eval(self, model, original_params, dx, dy,
                          alpha, beta, loader, device) -> float:
        """Set θ ← θ* + α·dx + β·dy and return mean loss."""
        with torch.no_grad():
            for p, w, d0, d1 in zip(model.parameters(), original_params, dx, dy):
                p.copy_(w.to(device) + alpha * d0.to(device) + beta * d1.to(device))
        return self._eval_loss(model, loader, device)

    def _restore_params(self, model, original_params):
        with torch.no_grad():
            for p, w in zip(model.parameters(), original_params):
                p.copy_(w)

    def _sweep_grid(self, model, original_params, dx, dy, loader, device):
        """Sweep the α×β grid and return (alphas, betas, loss_grid)."""
        alphas = np.linspace(-self.grid_range, self.grid_range, self.grid_points)
        betas  = np.linspace(-self.grid_range, self.grid_range, self.grid_points)
        loss_grid = np.zeros((self.grid_points, self.grid_points))

        n_params = sum(p.numel() for p in model.parameters())
        total_evals = self.grid_points * self.grid_points
        t0 = time.time()

        print(f"  grid:        {self.grid_points}×{self.grid_points}  ({total_evals} evaluations)", flush=True)
        print(f"  eval_batches per point: {self.eval_batches}", flush=True)
        print(f"  param count: {n_params:,}", flush=True)
        print(flush=True)

        loss_min = float("inf")
        loss_max = float("-inf")

        for i, alpha in enumerate(alphas):
            row_t0 = time.time()
            for j, beta in enumerate(betas):
                loss = self._perturb_and_eval(
                    model, original_params, dx, dy, alpha, beta, loader, device
                )
                loss_grid[i, j] = loss
                loss_min = min(loss_min, loss)
                loss_max = max(loss_max, loss)

            elapsed   = time.time() - t0
            row_time  = time.time() - row_t0
            frac      = (i + 1) / self.grid_points
            eta       = elapsed / frac - elapsed
            row_losses = loss_grid[i]
            print(
                f"  row {i+1:2d}/{self.grid_points}  "
                f"loss=[{row_losses.min():.4f}, {row_losses.max():.4f}]  "
                f"row={row_time:.1f}s  elapsed={elapsed:.1f}s  ETA={eta:.1f}s",
                flush=True,
            )

        print(f"\n  sweep done — overall loss range [{loss_min:.4f}, {loss_max:.4f}]", flush=True)
        return alphas, betas, loss_grid

    # ------------------------------------------------------------------ #
    # Plotting                                                             #
    # ------------------------------------------------------------------ #

    def _plot_surface(self, alphas, betas, loss_grid):
        print("  rendering static surface + contour …", flush=True)
        A, B = np.meshgrid(alphas, betas, indexing='ij')
        Z = loss_grid

        fig = plt.figure(figsize=(18, 7))
        fig.patch.set_facecolor('#0d0d0d')

        # 3-D surface
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.set_facecolor('#0d0d0d')
        surf = ax1.plot_surface(A, B, Z, cmap='inferno',
                                edgecolor='none', alpha=0.92,
                                rstride=1, cstride=1)
        ax1.set_title('Loss Landscape (3D Surface)', color='white', fontsize=13, pad=12)
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

        # 2-D contour
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_facecolor('#111111')
        levels = np.linspace(Z.min(), Z.max(), 25)
        cf = ax2.contourf(A, B, Z, levels=levels, cmap='inferno')
        ax2.contour(A, B, Z, levels=levels, colors='white', linewidths=0.3, alpha=0.4)
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
            f'{self.grid_points}×{self.grid_points} grid  |  '
            f'range ±{self.grid_range}  |  {self.eval_batches} eval batches/point',
            color='white', fontsize=11, y=1.01,
        )
        plt.tight_layout()
        out = os.path.join(self.save_dir, "loss_landscape.png")
        plt.savefig(out, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  saved → {out}")

    def _plot_1d_slices(self, alphas, betas, loss_grid):
        """1-D loss slices along each axis through θ* (centre row/col)."""
        print("  rendering 1-D slices …", flush=True)
        mid = self.grid_points // 2

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor('#0d0d0d')
        for ax in (ax1, ax2):
            ax.set_facecolor('#111111')
            ax.tick_params(colors='#888888')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333333')

        ax1.plot(alphas, loss_grid[:, mid], color='#ff6b35', linewidth=1.5)
        ax1.axvline(0, color='cyan', linewidth=0.8, linestyle='--')
        ax1.set_title('1-D slice along α (β=0)', color='white', fontsize=11)
        ax1.set_xlabel('α', color='#aaaaaa')
        ax1.set_ylabel('Loss', color='#aaaaaa')

        ax2.plot(betas, loss_grid[mid, :], color='mediumpurple', linewidth=1.5)
        ax2.axvline(0, color='cyan', linewidth=0.8, linestyle='--')
        ax2.set_title('1-D slice along β (α=0)', color='white', fontsize=11)
        ax2.set_xlabel('β', color='#aaaaaa')
        ax2.set_ylabel('Loss', color='#aaaaaa')

        plt.tight_layout()
        out = os.path.join(self.save_dir, "loss_landscape_1d_slices.png")
        plt.savefig(out, dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  saved → {out}")

    def _plot_interactive(self, alphas, betas, loss_grid):
        """Save a self-contained interactive HTML file with a rotatable 3D surface."""
        print("  rendering interactive 3D HTML …", flush=True)
        try:
            import plotly.graph_objects as go
        except ImportError:
            print("  warning: plotly not installed — skipping interactive visualization "
                  "(pip install plotly)")
            return

        A, B = np.meshgrid(alphas, betas, indexing='ij')
        Z = loss_grid

        fig = go.Figure(data=[
            go.Surface(
                x=A, y=B, z=Z,
                colorscale='Inferno',
                colorbar=dict(title='Loss', tickfont=dict(color='#aaaaaa')),
                opacity=0.92,
            )
        ])

        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[float(Z[self.grid_points // 2, self.grid_points // 2])],
            mode='markers',
            marker=dict(size=6, color='cyan', symbol='circle'),
            name='θ* (trained weights)',
        ))

        fig.update_layout(
            title=dict(
                text=(
                    f'Loss Landscape — filter-normalized random directions<br>'
                    f'{self.grid_points}×{self.grid_points} grid  |  '
                    f'range ±{self.grid_range}  |  {self.eval_batches} eval batches/point'
                ),
                font=dict(color='white', size=13),
            ),
            scene=dict(
                xaxis=dict(title='α (direction 1)', color='#aaaaaa',
                           backgroundcolor='#111111', gridcolor='#333333'),
                yaxis=dict(title='β (direction 2)', color='#aaaaaa',
                           backgroundcolor='#111111', gridcolor='#333333'),
                zaxis=dict(title='Loss',            color='#aaaaaa',
                           backgroundcolor='#111111', gridcolor='#333333'),
                bgcolor='#0d0d0d',
            ),
            paper_bgcolor='#0d0d0d',
            plot_bgcolor='#0d0d0d',
            font=dict(color='#aaaaaa'),
            legend=dict(font=dict(color='white')),
            margin=dict(l=0, r=0, t=60, b=0),
        )

        out = os.path.join(self.save_dir, "loss_landscape_3d.html")
        fig.write_html(out, include_plotlyjs='cdn')
        print(f"  saved → {out}")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def visualize(self, model, dataloader, device):
        """
        Run the full loss-landscape visualization.

        Snapshots model weights, sweeps an α×β grid of perturbations,
        restores the original weights, then saves plots and the raw
        loss grid to self.save_dir.

        Args:
            model:      nn.Module to analyze (must already be trained).
            dataloader: DataLoader supplying (inputs, targets) pairs.
            device:     torch.device to run inference on.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        model.to(device)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"[LossVisualizer] starting — {n_params:,} parameters", flush=True)
        print(f"  save_dir:    {self.save_dir}", flush=True)
        print(f"  grid:        {self.grid_points}×{self.grid_points}, range ±{self.grid_range}", flush=True)
        print(f"  eval_batches: {self.eval_batches}", flush=True)

        print("\n[LossVisualizer] snapshotting weights to CPU …", flush=True)
        original_params = [p.detach().cpu().clone() for p in model.parameters()]

        print("\n[LossVisualizer] generating filter-normalized random directions …", flush=True)
        dx = make_random_direction(original_params)
        dy = make_random_direction(original_params)

        print("\n[LossVisualizer] sweeping grid …", flush=True)
        t_sweep = time.time()
        alphas, betas, loss_grid = self._sweep_grid(
            model, original_params, dx, dy, dataloader, device
        )
        print(f"  total sweep time: {time.time() - t_sweep:.1f}s")

        print("\n[LossVisualizer] restoring original weights …", flush=True)
        self._restore_params(model, original_params)

        print("\n[LossVisualizer] saving plots …", flush=True)
        self._plot_surface(alphas, betas, loss_grid)
        self._plot_1d_slices(alphas, betas, loss_grid)
        if self.save_interactive_visualization:
            self._plot_interactive(alphas, betas, loss_grid)

        tensor_path = os.path.join(self.save_dir, "loss_grid.pth")
        torch.save({
            "alphas":    torch.tensor(alphas),
            "betas":     torch.tensor(betas),
            "loss_grid": torch.tensor(loss_grid),
        }, tensor_path)
        print(f"  saved → {tensor_path}")

        print("\n[LossVisualizer] done.", flush=True)
