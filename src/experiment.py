import json
from datetime import datetime
from pathlib import Path

import torch


class Experiment:

    def __init__(self, root: str = "artifacts", config: dict | None = None, tag: str = ""):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        label = f"{tag}_{timestamp}" if tag else timestamp
        self.path = Path(root) / label
        self.path.mkdir(parents=True, exist_ok=True)

        with open(self.path / "config.json", "w") as f:
            json.dump(config or {}, f, indent=4)

        self.dataset_path = self.path / "placements.jsonl"

    def append_sample(self, netlist_name: str, placement: dict, metrics: dict, failed: bool):
        sample = {
            "netlist": netlist_name,
            "placement": placement,
            "metrics": metrics,
            "failed": failed,
        }
        with open(self.dataset_path, "a") as f:
            json.dump(sample, f)
            f.write("\n")

    def save_model(self, model: torch.nn.Module, filename: str):
        torch.save(model.state_dict(), self.path / filename)

    def append_metrics(self, metrics: dict):
        metrics_path = self.path / "metrics.jsonl"
        with open(metrics_path, "a") as f:
            json.dump(metrics, f)
            f.write("\n")

