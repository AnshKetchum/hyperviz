# hyperviz

basic package to manage running visualizations done to better understand the inner workings of transformers.


```bash
from hyperviz import Visualizer 
from torch import MyTransformer

vis = Visualizer("./viz") # auto creates dir

net = MyTransformer()


for i in range(num_train_steps):
    x = torch.randn(10, 4096, 256)


    # hidden states is a list of all the hidden states, they need to be the same shape 
    # hidden states itself should be a list of (B, T, E) tensors
    pred, hidden_states = net(x)

    # compute loss ...

    # do a viz pass
    vis.add(hidden_states)

vis.visualize()


```