import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .environment import PlacementEnv
from .netlist import Netlist


def plot_placement(netlist: Netlist, config: dict, num_rows: int, num_cols: int, title: str = "", ax=None):

    env = PlacementEnv(netlist=netlist, num_rows=num_rows, num_cols=num_cols)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    canvas_width = netlist.canvas["width"]
    canvas_height = netlist.canvas["height"]

    for node_id, (row, col) in config.items():
        left, right, bottom, top = env.get_rectangle(node_id, row, col)

        rect = patches.Rectangle(
            (left, bottom),
            right - left,
            top - bottom,
            linewidth=1,
            edgecolor="black",
            facecolor="skyblue",
            alpha=0.7,
        )
        ax.add_patch(rect)

        # Label with node id at rectangle center
        cx, cy = (left + right) / 2, (bottom + top) / 2
        ax.text(cx, cy, node_id, ha="center", va="center", fontsize=7)

    # Draw net connections as thin lines between block centers
    for net_id, node_ids in netlist.nets.items():
        centers = []
        for node_id in node_ids:
            if node_id in config:
                row, col = config[node_id]
                left, right, bottom, top = env.get_rectangle(node_id, row, col)
                centers.append(((left + right) / 2, (bottom + top) / 2))

        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                ax.plot(
                    [centers[i][0], centers[j][0]],
                    [centers[i][1], centers[j][1]],
                    color="gray",
                    linewidth=0.4,
                    alpha=0.5,
                    zorder=0,
                )

    ax.set_xlim(0, canvas_width)
    ax.set_ylim(0, canvas_height)
    ax.set_aspect("equal")
    ax.set_title(title)

    return ax


def compare_placements(
    netlist: Netlist,
    config_before: dict,
    config_after: dict,
    num_rows: int,
    num_cols: int,
    hpwl_before: float | None = None,
    hpwl_after: float | None = None,
):

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
    plt.show()