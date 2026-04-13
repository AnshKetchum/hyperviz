import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class SpectralVisualizer:
    """
    Visualizes the spectral structure (singular values) of all 2-D weight
    matrices in a model.

    Saves to {save_directory}/spectral_values/:
      - spectral_distribution.png  — histogram of ALL singular values pooled
                                     across every weight matrix
      - <name>.png                 — per-matrix singular-value bar chart
    """

    def __init__(self, save_directory: str):
        self.save_dir = save_directory

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _collect_2d_weights(model: nn.Module) -> dict[str, torch.Tensor]:
        """Return {param_name: tensor} for every 2-D parameter in model."""
        return {
            name: param.detach().cpu()
            for name, param in model.named_parameters()
            if param.dim() == 2
        }

    @staticmethod
    def _singular_values(weight: torch.Tensor) -> torch.Tensor:
        """Compute singular values of a 2-D weight matrix (largest first)."""
        return torch.linalg.svdvals(weight.float())

    @staticmethod
    def _safe_filename(name: str) -> str:
        """Replace characters that are problematic in filenames."""
        return name.replace(".", "_").replace("/", "_")

    # ------------------------------------------------------------------ #
    # Plotting                                                             #
    # ------------------------------------------------------------------ #

    def _plot_global_distribution(self, all_svs: torch.Tensor, out_dir: str):
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#0d0d0d")
        ax.set_facecolor("#111111")

        vals = all_svs.numpy()
        ax.hist(vals, bins=120, color="#5b8dee", edgecolor="none", alpha=0.85)
        ax.axvline(float(vals.mean()), color="cyan", linewidth=1.2,
                   linestyle="--", label=f"mean={vals.mean():.3f}")
        ax.axvline(float(vals.max()),  color="#ff6b35", linewidth=1.0,
                   linestyle=":",  label=f"max={vals.max():.3f}")

        ax.set_title("Singular value distribution — all weight matrices",
                     color="white", fontsize=12)
        ax.set_xlabel("Singular value", color="#aaaaaa")
        ax.set_ylabel("Count", color="#aaaaaa")
        ax.tick_params(colors="#888888")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.legend(facecolor="#1a1a1a", edgecolor="#444444",
                  labelcolor="white", fontsize=9)

        plt.tight_layout()
        out = os.path.join(out_dir, "spectral_distribution.png")
        plt.savefig(out, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"  saved → {out}")

    def _plot_per_matrix(self, name: str, svs: torch.Tensor, out_dir: str):
        vals = svs.numpy()
        n = len(vals)

        fig, ax = plt.subplots(figsize=(max(6, n // 4), 4))
        fig.patch.set_facecolor("#0d0d0d")
        ax.set_facecolor("#111111")

        x = range(n)
        ax.bar(x, vals, color="#5b8dee", edgecolor="none", alpha=0.85)
        ax.axhline(float(vals.mean()), color="cyan", linewidth=1.0,
                   linestyle="--", label=f"mean={vals.mean():.3f}")

        ax.set_title(f"Singular values — {name}", color="white", fontsize=10)
        ax.set_xlabel("Index (largest → smallest)", color="#aaaaaa")
        ax.set_ylabel("Singular value", color="#aaaaaa")
        ax.tick_params(colors="#888888")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")
        ax.legend(facecolor="#1a1a1a", edgecolor="#444444",
                  labelcolor="white", fontsize=8)

        plt.tight_layout()
        fname = self._safe_filename(name) + ".png"
        out = os.path.join(out_dir, fname)
        plt.savefig(out, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def visualize(self, model: nn.Module):
        """
        Compute and plot singular values for all 2-D weight matrices in model.

        Args:
            model: nn.Module to analyze (weights are read without modifying them).
        """
        out_dir = os.path.join(self.save_dir, "spectral_values")
        os.makedirs(out_dir, exist_ok=True)

        weights = self._collect_2d_weights(model)
        if not weights:
            print("[SpectralVisualizer] no 2-D weight matrices found — skipping.")
            return

        print(f"[SpectralVisualizer] found {len(weights)} 2-D weight matrices")
        print(f"  save_dir: {out_dir}")

        all_svs = []
        for name, W in weights.items():
            svs = self._singular_values(W)
            all_svs.append(svs)
            self._plot_per_matrix(name, svs, out_dir)

        all_svs_cat = torch.cat(all_svs)
        self._plot_global_distribution(all_svs_cat, out_dir)

        # Save raw singular values
        tensor_path = os.path.join(out_dir, "singular_values.pth")
        torch.save({name: self._singular_values(W) for name, W in weights.items()},
                   tensor_path)
        print(f"  saved → {tensor_path}")

        print(f"[SpectralVisualizer] done — {len(weights)} matrices, "
              f"{len(all_svs_cat)} total singular values.")
