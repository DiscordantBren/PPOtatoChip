"""
Experiment runner: runs parameter sweeps with multiple seeds and generates PDF reports.
"""

import json
import random
import shutil
import threading
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from ...training import train_VanillaPVN, train_RewardPredictor, train_GraphPPO
from ...netlist import Netlist
from ...environment import PlacementEnv
from ...models import GraphPolicyValueNetwork
from ...visualize import plot_placement
import torch

A4 = (8.27, 11.69)


def run_experiment(
    base_config: dict,
    sweep_param: str,
    sweep_values: list,
    num_runs: int = 3,
    delete_artifacts: bool = False,
    stop_event: threading.Event | None = None,
    progress_callback=None,
) -> list[dict]:
    """Run a parameter sweep experiment with multiple runs per value."""
    results = []
    total = len(sweep_values) * num_runs
    run_counter = 0
    all_netlists = sorted([p.stem for p in Path("netlists").glob("*.json")])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for val_idx, val in enumerate(sweep_values):
        for run_idx in range(num_runs):
            if stop_event and stop_event.is_set():
                break

            seed = random.randint(0, 2**31 - 1)
            config = dict(base_config)
            config["mode"] = "full_pipeline"
            _apply_sweep(config, sweep_param, val)

            label = f"{sweep_param}={val} (run {run_idx+1})"
            np = config["netlist_path"]
            net_name = Path(np).stem

            if progress_callback:
                progress_callback({"kind": "run_start", "idx": run_counter, "total": total, "label": label,
                                   "val_idx": val_idx, "run_idx": run_idx})

            if progress_callback:
                progress_callback({"kind": "log", "text": f"  [{label}] Stage 1/3: Vanilla PVN (seed={seed})"})
            exp_vp = train_VanillaPVN(
                netlist_path=np, num_rows=config["num_rows"], num_cols=config["num_cols"],
                hidden_dim=config["hidden_dim"], num_hidden=config["num_hidden"],
                gamma=config["gamma"], clip_epsilon=config["clip_epsilon"],
                value_loss_coef=config["value_loss_coef"], entropy_coef=config["entropy_coef"],
                num_iterations=config.get("vanilla_iterations", 100), lr=config["lr"],
                stop_event=stop_event,
            )
            if stop_event and stop_event.is_set():
                break

            if progress_callback:
                progress_callback({"kind": "log", "text": f"  [{label}] Stage 2/3: Reward Predictor"})
            netlists = {net_name: Netlist(np)}
            rp = config.get("reward_predictor", {})
            _, exp_rp = train_RewardPredictor(
                placements_path=str(exp_vp.path / "placements.jsonl"),
                netlists=netlists, num_rows=config["num_rows"], num_cols=config["num_cols"],
                hidden_channels_e=config["hidden_channels_e"], num_layers_e=config["num_layers_e"],
                hidden_channels_r=rp.get("hidden_channels_r", 128),
                num_layers_r=rp.get("num_layers_r", 3),
                batch_size=rp.get("batch_size", 32), num_epochs=rp.get("num_epochs", 100),
                lr=config["lr"], stop_event=stop_event,
            )
            if stop_event and stop_event.is_set():
                break

            if progress_callback:
                progress_callback({"kind": "log", "text": f"  [{label}] Stage 3/3: Graph PPO"})
            pp = config.get("graph_ppo", {})
            encoder_path = exp_rp.path / "encoder.pt"
            model, exp = train_GraphPPO(
                netlist_path=np, num_rows=config["num_rows"], num_cols=config["num_cols"],
                hidden_channels_e=config["hidden_channels_e"], num_layers_e=config["num_layers_e"],
                hidden_dim=config["hidden_dim"], num_hidden=config["num_hidden"],
                pretrained_encoder_path=str(encoder_path),
                freeze_encoder=pp.get("freeze_encoder", False),
                gamma=config["gamma"], clip_epsilon=config["clip_epsilon"],
                value_loss_coef=config["value_loss_coef"], entropy_coef=config["entropy_coef"],
                num_iterations=pp.get("num_iterations", 100), lr=config["lr"],
                stop_event=stop_event,
            )
            if stop_event and stop_event.is_set():
                break

            hpwl = _extract_hpwl(exp.path)
            hpwl_vp = _extract_hpwl(exp_vp.path)

            # Evaluate on all netlists for scale generalization
            if progress_callback:
                progress_callback({"kind": "log", "text": f"  [{label}] Evaluating on all netlists..."})
            netlist_hpwls = {}
            with torch.no_grad():
                for nl in all_netlists:
                    nlist = Netlist(f"netlists/{nl}.json")
                    env = PlacementEnv(netlist=nlist, num_rows=config["num_rows"], num_cols=config["num_cols"])
                    env.reset()
                    done, failed = False, False
                    while not done:
                        go = env.get_graph_observation().to(device)
                        am = torch.as_tensor(env.get_action_mask(), dtype=torch.bool, device=device)
                        logits, _ = model(go.x, go.edge_index, go.edge_weight, go.current_node_idx)
                        masked = logits.squeeze(0).masked_fill(~am, -torch.inf)
                        action = torch.argmax(masked).item()
                        _, done, failed = env.step(int(action))
                    if not failed:
                        netlist_hpwls[nl] = env.get_metrics()["hpwl"]

            result = {
                "value": val, "seed": seed, "label": label,
                "hpwl": hpwl, "hpwl_vp": hpwl_vp,
                "run_path": str(exp.path), "vp_path": str(exp_vp.path), "rp_path": str(exp_rp.path),
                "netlist": net_name,
                "netlist_hpwls": netlist_hpwls,
            }
            results.append(result)
            run_counter += 1

            if progress_callback:
                progress_callback({"kind": "run_done", "idx": run_counter - 1, "total": total,
                                   "label": label, "hpwl": hpwl, "val_idx": val_idx, "run_idx": run_idx})

        if stop_event and stop_event.is_set():
            break

    if results and not (stop_event and stop_event.is_set()):
        report_path = _generate_report(results, sweep_param, sweep_values, num_runs, all_netlists)
        for r in results:
            r["report_path"] = str(report_path)

        if delete_artifacts:
            _delete_run_artifacts(results, progress_callback)

    return results


def _extract_hpwl(exp_path: Path) -> float | None:
    metrics_path = exp_path / "metrics.jsonl"
    if metrics_path.exists():
        with open(metrics_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        successful = [l for l in lines if not l.get("failed", False) and l.get("hpwl") is not None]
        if successful:
            return min(s["hpwl"] for s in successful)
    return None


def _delete_run_artifacts(results, progress_callback=None):
    paths = set()
    for r in results:
        paths.add(r.get("run_path"))
        paths.add(r.get("vp_path"))
        paths.add(r.get("rp_path"))
    for p in paths:
        if p:
            path = Path(p)
            if path.exists():
                shutil.rmtree(path)
                if progress_callback:
                    progress_callback({"kind": "log", "text": f"  Deleted: {path.name}"})


def _apply_sweep(config: dict, param: str, val) -> None:
    if param in ("gamma", "clip_epsilon", "value_loss_coef", "entropy_coef", "lr",
                 "num_iterations", "num_rows", "num_cols", "hidden_dim", "num_hidden",
                 "hidden_channels_e", "num_layers_e"):
        config[param] = val
        if param in ("hidden_channels_e", "num_layers_e"):
            config.setdefault("reward_predictor", {})[param] = val
            config.setdefault("graph_ppo", {})[param] = val
    elif param == "freeze_encoder":
        config.setdefault("graph_ppo", {})["freeze_encoder"] = val
    elif param in ("batch_size", "num_epochs", "hidden_channels_r", "num_layers_r"):
        config.setdefault("reward_predictor", {})[param] = val


def _median_run(val_results):
    sorted_r = sorted(val_results, key=lambda r: r["hpwl"] if r["hpwl"] is not None else float("inf"))
    return sorted_r[len(sorted_r) // 2]


def _load_placement_config(run_path: Path):
    pp = Path(run_path) / "placements.jsonl"
    if pp.exists():
        samples = [json.loads(l) for l in open(pp) if l.strip()]
        success = [s for s in samples if not s.get("failed", False) and "placement" in s]
        if success:
            best = min(success, key=lambda s: s["metrics"]["hpwl"])
            return {nid: tuple(coords) for nid, coords in best["placement"].items()}
    return None


# ── Report Generator ──

def _generate_report(results, sweep_param, sweep_values, num_runs, all_netlists) -> Path:
    exp_dir = Path("analysis") / "experiments" / datetime.now().strftime("exp_%Y-%m-%d_%H-%M-%S")
    exp_dir.mkdir(parents=True, exist_ok=True)
    report_path = exp_dir / "report.pdf"

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
              "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    val_colors = {v: colors[i % len(colors)] for i, v in enumerate(sweep_values)}

    json.dump({
        "sweep_param": sweep_param, "sweep_values": sweep_values, "num_runs": num_runs,
        "results": [{"value": r["value"], "seed": r.get("seed"), "hpwl": r["hpwl"],
                      "run_path": r["run_path"]} for r in results],
    }, open(exp_dir / "config.json", "w"), indent=2, default=str)

    with PdfPages(report_path) as pdf:
        _page_vanilla_vs_trained(pdf, results, sweep_param, sweep_values, val_colors)
        _page_netlist_bars(pdf, results, sweep_param, sweep_values, val_colors, num_runs, all_netlists)
        _page_vanilla_vs_trained_placements(pdf, results, sweep_param, sweep_values)
        _page_netlist_placements(pdf, results, sweep_param, sweep_values, all_netlists, val_colors)
        _page_learning_curves(pdf, results, sweep_param, sweep_values, val_colors)
        _page_summary_table(pdf, results, sweep_param, sweep_values, num_runs)

    print(f"Report saved: {report_path}")
    return report_path


# ── Page 1: Vanilla vs Trained HPWL ──

def _page_vanilla_vs_trained(pdf, results, sweep_param, sweep_values, val_colors):
    fig, ax = plt.subplots(figsize=A4)
    for val in sweep_values:
        val_results = [r for r in results if r["value"] == val]
        color = val_colors[val]
        for vr in val_results:
            vp = vr.get("hpwl_vp")
            gp = vr.get("hpwl")
            if vp is not None and gp is not None:
                ax.plot([0, 1], [vp, gp], color=color, alpha=0.3, linewidth=0.5, marker="o")
        vps = [r["hpwl_vp"] for r in val_results if r.get("hpwl_vp") is not None]
        gps = [r["hpwl"] for r in val_results if r.get("hpwl") is not None]
        if vps and gps:
            ax.plot([0, 1], [sum(vps)/len(vps), sum(gps)/len(gps)], color=color, linewidth=2,
                    marker="s", markersize=8, label=f"{sweep_param}={val} (mean)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Vanilla PVN", "Graph PPO"])
    ax.set_ylabel("HPWL")
    ax.set_title("Vanilla PVN vs Graph PPO HPWL (all runs)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


# ── Page 2: HPWL per netlist, grouped bars ──

def _page_netlist_bars(pdf, results, sweep_param, sweep_values, val_colors, num_runs, all_netlists):
    fig, ax = plt.subplots(figsize=A4)
    n_groups = len(all_netlists)
    n_bars = len(sweep_values)
    x = list(range(n_groups))
    width = 0.8 / n_bars

    for bi, val in enumerate(sweep_values):
        means = []
        ranges = []
        for net_name in all_netlists:
            hpwls = []
            for r in results:
                if r["value"] == val:
                    nh = r.get("netlist_hpwls", {})
                    if net_name in nh and nh[net_name] is not None:
                        hpwls.append(nh[net_name])
            if hpwls:
                means.append(sum(hpwls) / len(hpwls))
                ranges.append(max(hpwls) - min(hpwls) if len(hpwls) > 1 else 0)
            else:
                means.append(0)
                ranges.append(0)
        offset = (bi - n_bars / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], means, width, yerr=ranges,
               color=val_colors[val], edgecolor="black", capsize=3,
               label=f"{sweep_param}={val}")

    ax.set_xticks(x)
    ax.set_xticklabels(all_netlists, rotation=45, ha="right")
    ax.set_ylabel("HPWL")
    ax.set_title(f"HPWL by Netlist (grouped by {sweep_param}, {num_runs} runs each)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


# ── Page 3: Vanilla vs Trained placements (median run per sweep value) ──

def _page_vanilla_vs_trained_placements(pdf, results, sweep_param, sweep_values):
    valid = [r for r in results if r["hpwl"] is not None]
    if not valid:
        return

    netlist_name = valid[0].get("netlist", "xerox")
    netlist = Netlist(f"netlists/{netlist_name}.json")

    rows_per_page = 3
    n = len(sweep_values)
    total_pages = (n + rows_per_page - 1) // rows_per_page

    for page in range(total_pages):
        start = page * rows_per_page
        end = min(start + rows_per_page, n)
        page_rows = end - start

        fig, axes = plt.subplots(page_rows, 2, figsize=(A4[0], A4[1] * 0.4 * page_rows))
        if page_rows == 1:
            axes = [axes]  # axes is a 1D array of 2 elements: [ax_left, ax_right]

        for i in range(start, end):
            val = sweep_values[i]
            val_results = [r for r in results if r["value"] == val and r["hpwl"] is not None]
            if not val_results:
                continue
            mr = _median_run(val_results)
            ri = i - start

            # When page_rows == 1, axes is [array_of_2_axes], so axes[ri] is the whole array
            # We need axes[ri] to be the left/right pair
            if page_rows == 1:
                vp_ax = axes[0][0]
                gp_ax = axes[0][1]
            else:
                vp_ax = axes[ri][0]
                gp_ax = axes[ri][1]

            vp_config = _load_placement_config(Path(mr["vp_path"]))
            if vp_config:
                plot_placement(netlist, vp_config, 128, 128,
                              title=f"Vanilla PVN ({sweep_param}={val}, HPWL: {mr.get('hpwl_vp', '?'):.2f})", ax=vp_ax)
            else:
                vp_ax.text(0.5, 0.5, f"Vanilla ({sweep_param}={val})\nNo data", ha="center", va="center", transform=vp_ax.transAxes)

            gp_config = _load_placement_config(Path(mr["run_path"]))
            if gp_config:
                plot_placement(netlist, gp_config, 128, 128,
                              title=f"Graph PPO ({sweep_param}={val}, HPWL: {mr['hpwl']:.2f})", ax=gp_ax)
            else:
                gp_ax.text(0.5, 0.5, f"Graph PPO ({sweep_param}={val})\nNo data", ha="center", va="center", transform=gp_ax.transAxes)

        fig.suptitle(f"Vanilla PVN vs Graph PPO (median run, page {page+1}/{total_pages})", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)


# ── Page 4+: One page per netlist, placements for each sweep value ──

def _page_netlist_placements(pdf, results, sweep_param, sweep_values, all_netlists, val_colors):
    train_netlist = results[0]["netlist"] if results else "xerox"
    show_netlists = [train_netlist] + [n for n in all_netlists if n != train_netlist][:4]
    n_vals = len(sweep_values)
    if n_vals == 0:
        return

    for net_name in show_netlists:
        plots_per_page = 6
        total_pages = (n_vals + plots_per_page - 1) // plots_per_page

        for page in range(total_pages):
            start = page * plots_per_page
            end = min(start + plots_per_page, n_vals)
            page_vals = end - start
            cols = min(3, page_vals)
            rows = (page_vals + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(A4[0], A4[1] * 0.4 * rows))
            # Flatten to 1D for simple indexing
            axes_flat = axes.flatten() if rows > 1 or cols > 1 else [axes]

            for i in range(start, end):
                val = sweep_values[i]
                ax = axes_flat[i - start]

                val_results = [r for r in results if r["value"] == val and r["hpwl"] is not None]
                if not val_results:
                    ax.text(0.5, 0.5, f"{sweep_param}={val}\nNo data", ha="center", va="center", transform=ax.transAxes)
                    continue

                mr = _median_run(val_results)
                model, device = _load_model(Path(mr["run_path"]))
                nlist = Netlist(f"netlists/{net_name}.json")
                env = PlacementEnv(netlist=nlist, num_rows=128, num_cols=128)
                env.reset()
                done, failed = False, False
                while not done:
                    go = env.get_graph_observation().to(device)
                    am = torch.as_tensor(env.get_action_mask(), dtype=torch.bool, device=device)
                    with torch.no_grad():
                        logits, _ = model(go.x, go.edge_index, go.edge_weight, go.current_node_idx)
                        masked = logits.squeeze(0).masked_fill(~am, -torch.inf)
                        action = torch.argmax(masked).item()
                    _, done, failed = env.step(int(action))
                if not failed:
                    config = dict(env.config)
                    hpwl = env.get_metrics()["hpwl"]
                    plot_placement(nlist, config, 128, 128,
                                  title=f"{sweep_param}={val}\nHPWL: {hpwl:.0f}", ax=ax)
                else:
                    ax.text(0.5, 0.5, f"{sweep_param}={val}\nFailed", ha="center", va="center", transform=ax.transAxes)

            for i in range(page_vals, len(axes_flat)):
                axes_flat[i].set_visible(False)

            fig.suptitle(f"Placements on {net_name} across {sweep_param} (page {page+1}/{total_pages})", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)


# ── Learning Curves ──

def _page_learning_curves(pdf, results, sweep_param, sweep_values, val_colors):
    fig, ax = plt.subplots(figsize=A4)
    for val in sweep_values:
        val_results = [r for r in results if r["value"] == val]
        color = val_colors[val]
        all_hpwls = []
        for vr in val_results:
            mp = Path(vr["run_path"]) / "metrics.jsonl"
            if mp.exists():
                data = [json.loads(l) for l in open(mp) if l.strip()]
                series = [d["hpwl"] for d in data if not d.get("failed", False) and d.get("hpwl") is not None]
                if series:
                    ax.plot(range(len(series)), series, marker=".", linestyle="-", color=color, alpha=0.3, linewidth=0.5)
                    all_hpwls.append(series)
        if all_hpwls:
            min_len = min(len(h) for h in all_hpwls)
            mean_s = [sum(h[i] for h in all_hpwls) / len(all_hpwls) for i in range(min_len)]
            ax.plot(range(min_len), mean_s, linestyle="-", color=color, linewidth=2, label=f"{sweep_param}={val} (mean)")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("HPWL")
    ax.set_title(f"Learning Curves: {sweep_param} Sweep")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


# ── Summary Table ──

def _page_summary_table(pdf, results, sweep_param, sweep_values, num_runs):
    fig, ax = plt.subplots(figsize=A4)
    ax.axis("off")

    headers = [sweep_param, "Run", "Seed", "HPWL", "Vanilla HPWL"]
    rows = [headers]

    for val in sweep_values:
        val_results = [r for r in results if r["value"] == val]
        for i, vr in enumerate(val_results):
            rows.append([
                str(val) if i == 0 else "",
                str(i + 1),
                str(vr.get("seed", "?")),
                f"{vr['hpwl']:.2f}" if vr["hpwl"] else "N/A",
                f"{vr.get('hpwl_vp', 0):.2f}" if vr.get("hpwl_vp") else "N/A",
            ])
        hpwls = [r["hpwl"] for r in val_results if r["hpwl"] is not None]
        if hpwls:
            rows.append(["", "Mean", "", f"{sum(hpwls)/len(hpwls):.2f}", ""])
        rows.append(["", "", "", "", ""])

    table = ax.table(cellText=rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.3)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#333333")
            cell.set_text_props(color="white")
    ax.set_title(f"Experiment Results: {sweep_param} Sweep ({num_runs} runs each)", fontsize=12, pad=20)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def _load_model(run_path: Path, num_rows=128, num_cols=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(run_path / "config.json") as f:
        cfg = json.load(f)
    dummy = Netlist("netlists/xerox.json")
    de = PlacementEnv(netlist=dummy, num_rows=num_rows, num_cols=num_cols)
    do = de.get_graph_observation()
    assert do.x is not None
    model = GraphPolicyValueNetwork(
        output_dim=num_rows * num_cols, in_channels=do.x.shape[1],
        hidden_channels_e=cfg.get("hidden_channels_e", 128),
        num_layers_e=cfg.get("num_layers_e", 3),
        hidden_dim=cfg.get("hidden_dim", 128),
        num_hidden=cfg.get("num_hidden", 3),
    ).to(device)
    ckpt = run_path / "graph_ppo_final.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model, device