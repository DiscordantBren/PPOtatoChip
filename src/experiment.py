from __future__ import annotations  

import json
from datetime import datetime
from pathlib import Path

import torch

# Manages all artifacts produced during a single training run.
class Experiment:

    # Create timestamped experiment directory
    def __init__(self, root: str = "artifacts", config: dict | None = None):
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = Path(root) / timestamp
        self.path.mkdir(parents=True, exist_ok=False)

        # config file
        with open(self.path / "config.json", "w") as f:
            json.dump(config or {}, f, indent=4)

        # Placement dataset file
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

    # Save a PyTorch model checkpoint.
    def save_model(self, model: torch.nn.Module, filename: str):

        torch.save(model.state_dict(), self.path / filename)

