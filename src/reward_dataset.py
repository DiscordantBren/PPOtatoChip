import json

import torch
from torch_geometric.data import Dataset
from torch_geometric.data import Data

from .netlist import Netlist


class RewardDataSet(Dataset):

    def __init__(self, placements_path: str, netlists: dict[str, Netlist], num_rows: int, num_cols: int) -> None:
        super().__init__()

        self.netlists = netlists
        self.num_rows = num_rows
        self.num_cols = num_cols
        
        self.samples = []
        
        with open(placements_path, "r") as f:
            for line in f:
                self.samples.append(json.loads(line))


    def len(self) -> int:
        return len(self.samples)

    def get(self, idx: int) -> Data:

        sample = self.samples[idx]
        netlist = self.netlists[sample["netlist"]]
        placement = sample["placement"]

        # Making the Node features
        node_features = []
        
        canvas_width = netlist.canvas["width"]
        canvas_height = netlist.canvas["height"]
        
        for node_id in list(netlist.nodes.keys()):
        
            width = netlist.nodes[node_id]["width"] / canvas_width
            height = netlist.nodes[node_id]["height"] / canvas_height
        
        
            row, col = placement[node_id]
        
            x_grid = (col + 0.5) / self.num_cols
            y_grid = (row + 0.5) / self.num_rows
        
            node_features.append([width, height, x_grid, y_grid, 1.0])
        
        x = torch.tensor(node_features, dtype=torch.float32)

        # Making the edge_index and edge_weight
        edges = []
        weights = [] 

        node_to_idx = {node_id:idx for idx, node_id in enumerate(list(netlist.nodes.keys()))}

        for blocks in netlist.nets.values():

            block_list = list(blocks)

            for i in range(len(block_list)):
                for j in range(i+1, len(block_list)):
                    if (node_to_idx[block_list[i]], node_to_idx[block_list[j]]) in edges:
                        weights[edges.index((node_to_idx[block_list[i]], node_to_idx[block_list[j]]))] += 1
                    else:
                        edges.append((node_to_idx[block_list[i]], node_to_idx[block_list[j]]))
                        weights.append(1)

        # Accounting for the undirected nature of the graph
        edges_bi = []
        weights_bi = []

        for (u,v), w in zip(edges, weights):
            edges_bi.append([u, v])
            edges_bi.append([v, u])
            weights_bi.append(w)
            weights_bi.append(w)

        edge_index = torch.tensor(edges_bi, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(weights_bi, dtype=torch.float32)

        # True reward as Graph label
        y = torch.tensor([sample["metrics"]["hpwl"] / (canvas_height + canvas_width)], dtype=torch.float32)
        
        return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)