"""
Matplotlib plot generators for PPOtatoChip analysis.
Each function is self-contained — no CLI logic, just data → figure.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from ..netlist import Netlist
from ..environment import PlacementEnv
from ..models import GraphPolicyValueNetwork
from ..visualize import plot_placement


def learning_curve(run_path: Path) -> plt.Figure:
    """Plot HPWL and PPO loss components over training iterations."""

    metrics_path = run_path / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics.jsonl found in {run_path}")

    data = []
    with open(metrics_path) as f:
        for line in f:
            data.append(json.loads(line))

    iterations = [d["iteration"] for d in data if not d.get("failed", False)]
    hpwls = [d["hpwl"] for d in data if not d.get("failed", False) and d.get("hpwl") is not None]
    rewards = [d["reward"] for d in data]
    policy_losses = [d["policy_loss_mean"] for d in data]
    value_losses = [d["value_loss_mean"] for d in data]
    entropies = [d["entropy_mean"] for d in data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: HPWL and Total Reward
    ax = axes[0]
    if hpwls:
        ax.plot(iterations, hpwls, marker="o", linestyle="-", label="HPWL", color="tab:blue")
    if rewards:
        ax.plot(range(len(rewards)), rewards, marker="s", linestyle="--", label="Reward", color="tab:orange")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.set_title("Placement Quality over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: PPO Loss Components
    ax = axes[1]
    ax.plot(range(len(policy_losses)), policy_losses, label="Policy Loss", alpha=0.8)
    ax.plot(range(len(value_losses)), value_losses, label="Value Loss", alpha=0.8)
    ax.plot(range(len(entropies)), entropies, label="Entropy", alpha=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("PPO Training Signals")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Run: {run_path.name}", fontsize=12, y=1.02)
    fig.tight_layout()
    return fig


def placement_compare(
    before_run_path: Path,
    after_run_path: Path,
    netlist_name: str,
    num_rows: int,
    num_cols: int,
) -> plt.Figure:
    """Side-by-side placement comparison between two runs (e.g. vanilla PVN vs Graph PPO)."""

    def _best_placement(run_path: Path) -> tuple[dict, float]:
        samples_path = run_path / "placements.jsonl"
        if not samples_path.exists():
            raise FileNotFoundError(f"No placements.jsonl found in {run_path}")

        samples = []
        with open(samples_path) as f:
            for line in f:
                samples.append(json.loads(line))

        matching = [
            s for s in samples
            if s["netlist"] == netlist_name and not s["failed"]
        ]
        if not matching:
            raise ValueError(
                f"No successful placements for '{netlist_name}' in {run_path.name}"
            )

        best = min(matching, key=lambda s: s["metrics"]["hpwl"])
        config = {
            node_id: tuple(coords) for node_id, coords in best["placement"].items()
        }
        hpwl = best["metrics"]["hpwl"]
        return config, hpwl

    config_before, hpwl_before = _best_placement(before_run_path)
    config_after, hpwl_after = _best_placement(after_run_path)

    # Find the netlist path from one of the configs
    with open(before_run_path / "config.json") as f:
        config_data = json.load(f)
    netlist_path = config_data.get("netlist_path", f"netlists/{netlist_name}.json")
    netlist = Netlist(netlist_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    title_before = "Vanilla PVN Placement"
    if hpwl_before is not None:
        title_before += f" (HPWL: {hpwl_before:.2f})"

    title_after = "Trained Graph PPO Placement"
    if hpwl_after is not None:
        title_after += f" (HPWL: {hpwl_after:.2f})"

    plot_placement(netlist, config_before, num_rows, num_cols, title=title_before, ax=ax1)
    plot_placement(netlist, config_after, num_rows, num_cols, title=title_after, ax=ax2)

    plt.tight_layout()
    return fig


def scale_generalize(
    model_run_path: Path,
    netlist_paths: list[str],
    netlist_names: list[str],
    num_rows: int,
    num_cols: int,
) -> plt.Figure:
    """Evaluate a trained Graph PPO model across multiple netlists and generate placement visualizations."""

    # Load config to get architecture dimensions
    with open(model_run_path / "config.json") as f:
        train_config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model from one netlist to infer input dims, then swap netlists per eval
    dummy_netlist = Netlist(netlist_paths[0])
    dummy_env = PlacementEnv(netlist=dummy_netlist, num_rows=num_rows, num_cols=num_cols)
    dummy_obs = dummy_env.get_graph_observation()
    assert dummy_obs.x is not None
    in_channels = dummy_obs.x.shape[1]
    num_actions = num_rows * num_cols

    model = GraphPolicyValueNetwork(
        output_dim=num_actions,
        in_channels=in_channels,
        hidden_channels_e=train_config.get("hidden_channels_e", 128),
        num_layers_e=train_config.get("num_layers_e", 3),
        hidden_dim=train_config.get("hidden_dim", 128),
        num_hidden=train_config.get("num_hidden", 3),
    ).to(device)

    checkpoint_path = model_run_path / "graph_ppo_final.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No graph_ppo_final.pt found in {model_run_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    def _deterministic_rollout(env):
        env.reset()
        done, failed = False, False
        while not done:
            graph_obs = env.get_graph_observation().to(device)
            action_mask = torch.as_tensor(
                env.get_action_mask(), dtype=torch.bool, device=device
            )
            with torch.no_grad():
                logits, _ = model(
                    graph_obs.x,
                    graph_obs.edge_index,
                    graph_obs.edge_weight,
                    graph_obs.current_node_idx,
                )
                masked_logits = logits.squeeze(0).masked_fill(~action_mask, -torch.inf)
                action = torch.argmax(masked_logits).item()
            _, done, failed = env.step(int(action))
        if failed:
            return None, None
        return dict(env.config), env.get_metrics()["hpwl"]

    placements = []
    hpwls = []
    for net_path, net_name in zip(netlist_paths, netlist_names):
        netlist = Netlist(net_path)
        env = PlacementEnv(netlist=netlist, num_rows=num_rows, num_cols=num_cols)
        config, hpwl = _deterministic_rollout(env)
        placements.append((netlist, config, net_name, hpwl))
        hpwls.append(hpwl)

    # Determine grid layout
    n = len(placements)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (netlist, config, name, hpwl) in enumerate(placements):
        ax = axes[i]
        if config is not None:
            from ..visualize import plot_placement
            title = f"{name} (HPWL: {hpwl:.2f})"
            plot_placement(netlist, config, num_rows, num_cols, title=title, ax=ax)
        else:
            ax.text(0.5, 0.5, f"{name}\nFailed", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name)

    # Hide unused subplots
    for i in range(n, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Scale Generalization — Model: {model_run_path.name}", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def encoder_ablation(
    run_paths: list[Path],
    labels: list[str],
) -> plt.Figure:
    """Overlay learning curves from multiple Graph PPO runs (frozen vs fine-tuned vs scratch)."""

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for run_path, label, color in zip(run_paths, labels, colors):
        metrics_path = run_path / "metrics.jsonl"
        if not metrics_path.exists():
            print(f"  [WARN] No metrics.jsonl in {run_path.name}, skipping")
            continue

        data = []
        with open(metrics_path) as f:
            for line in f:
                data.append(json.loads(line))

        hpwls = [d["hpwl"] for d in data if not d.get("failed", False) and d.get("hpwl") is not None]

        if hpwls:
            ax.plot(
                range(len(hpwls)),
                hpwls,
                marker=".",
                linestyle="-",
                label=label,
                color=color,
                alpha=0.85,
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("HPWL")
    ax.set_title("Encoder Ablation: Pretrained vs Frozen vs Scratch")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig