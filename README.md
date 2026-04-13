# hyperviz

basic package to manage running visualizations done to better understand the inner workings of transformers.

current features:

- Implementation of https://arxiv.org/abs/2602.05970 (inverse depth scaling visualizations)
- Implementation of loss-landscape visualization techniques (https://arxiv.org/pdf/1712.09913)


## Hidden-state visualizer (`Visualizer`)

Collects hidden-state trajectories across training and produces angle, delta-angle, relative-norm, and PCA plots.

```python
from hyperviz import Visualizer

vis = Visualizer("./viz")   # auto-creates dir on visualize()

net = MyTransformer()

for i in range(num_train_steps):
    x = torch.randn(10, 4096, 256)

    # hidden_states: list of (B, T, E) tensors, one per layer
    pred, hidden_states = net(x)

    # compute loss, step optimizer ...

    vis.add(hidden_states)

vis.visualize()
# saves: hidden_state_angle.png, hidden_state_delta_angle.png,
#        hidden_state_rel_norm.png, hidden_state_pca.png,
#        visualization_tensors.pth
```

## Loss-landscape visualizer (`LossVisualizer`)

Perturbs model weights along two filter-normalized random directions (Li et al., 2018) and plots the resulting loss surface. All hyperparameters are fixed at init time.

```python
from hyperviz.loss_visualizer import LossVisualizer
import torch.nn as nn

# train your model first ...

vis = LossVisualizer(
    save_directory="./loss_viz",
    criterion=nn.CrossEntropyLoss(),       # default
    grid_points=20,                        # N×N grid (default 20)
    grid_range=1.0,                        # sweep ±grid_range (default 1.0)
    eval_batches=50,                       # batches per grid point (default 50)
    save_interactive_visualization=True,   # also save a rotatable 3D HTML (requires plotly)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vis.visualize(model, dataloader, device)
# saves: loss_landscape.png              (3D surface + 2D contour, static)
#        loss_landscape_1d_slices.png    (cross-sections through θ*)
#        loss_landscape_3d.html          (interactive, rotate/zoom in browser)
#        loss_grid.pth                   (raw alphas / betas / loss_grid tensors)
```

### Low-level API (`loss_landscape.py`)

The underlying functions are also importable directly:

```python
from hyperviz.loss_landscape import (
    make_random_direction,   # filter-normalized random direction
    perturb_and_eval,        # single-point loss evaluation
    compute_landscape,       # sweep full α×β grid
    plot_landscape,          # render and save plots
)

params_cpu = [p.detach().cpu() for p in model.parameters()]
dx = make_random_direction(params_cpu)
dy = make_random_direction(params_cpu)

alphas, betas, loss_grid = compute_landscape(
    model, params_cpu, loader, device,
    criterion=nn.CrossEntropyLoss(),
    grid_range=1.0,
    grid_points=20,
    eval_batches=50,
)

plot_landscape(alphas, betas, loss_grid, out_dir="./loss_viz")
```