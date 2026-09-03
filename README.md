# PPOtatoChip

AlphaChip, but less alpha.
Work in progress.

Based on the Google AlphaChip paper - Chip Placement with Deep Reinforcement Learning: arxiv/2004.10746, 2020.

## Screenshots

![TUI](tui_img.png)

-- The TUI.

<br>
<br>
![Demo placement for ami49](demo_placement.png)

-- A demo placement by the model fine-tuned on the 10-component xerox netlist and applied on the 49-component ami49 netlist.
<br>
<br>
<br>

## Pipeline

The idea is to sequentially place components on a canvas while keeping wirelength low.

There are three stages in the training pipeline:

1. **Vanilla PVN** — a policy-value network sequentially place components, using negative HPWL (Half-Perimeter Wirelength) as the reward. Once a placement is complete, Proximal Policy Optimization (PPO) is used to update the network. The placements and their HPWL values are collected into a dataset.
2. **Reward Predictor** — trains an MLP attached to a Graph Neural Network (GNN) to predict wirelength of a placement using the dataset generated above. The GNN captures the structural relationships between components in the embedding.
3. **Graph PPO** — the pretrained GNN encoder is attached to a policy-value network and training is done using PPO.

The trained model is agnostic of the number of components to be placed and can be applied to a different netlist to generate a placement.


## Features
1. Pick a netlist, tweak the parameters, and start training. Training can be cancelled mid-run.

2. There is an **experiment system** for parameter sweeps (learning rate,
model capacity etc.). It runs multiple seeds and
generates a multi-page PDF report with learning curves, HPWL comparisons,
side-by-side placement grids, and a summary table.

3. The `netlists/` directory currently contains five netlists: `xerox`, `ami33`, `ami49`, `apte`, and `hp`. Training outputs are saved to `artifacts/` with timestamps, while plots and analysis are saved to `analysis/`.

## Operation

Dependencies:
```
torch
torch_geometric
textual
matplotlib
```

Clone the repo, install the dependencies, and run the TUI from inside the PPOtatoChip directory:

```bash
python src/cli.py
```



## Planned Updates

- **Convergence analysis** — figure out when training converges.
- **Better wirelength metric** — HPWL is convenient, but doesn't capture routing 
quality very well. Try Steiner-tree based metrics and also account for congestion.
- **Hybrid with classical placement** — try combining RL with Simulated Annealing or Force-directed Placement. AlphaChip uses RL for macros and conventional placers for standard cells.