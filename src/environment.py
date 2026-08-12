import torch
from torch_geometric.data import Data

from .netlist import Netlist
from .reward import compute_hpwl


def compute_placement_order(netlist: Netlist) -> list[str]:
    # Group node IDs by area
    nodes_by_area = {}

    for node_id in netlist.nodes:
        area = netlist.nodes[node_id]["width"] * netlist.nodes[node_id]["height"]

        if area not in nodes_by_area:
            nodes_by_area[area] = []

        nodes_by_area[area].append(node_id)

    # Process area groups from largest to smallest
    sorted_areas = sorted(nodes_by_area.keys(), reverse=True)

    # Each are group is ordered internally based on no. of connection to previous blocks
    ordered = []
    
    for area in sorted_areas:
        pending_nodes = nodes_by_area[area]

        if len(pending_nodes) == 1:
            ordered.append(pending_nodes.pop())
        else:
            while len(pending_nodes) > 0:
                scores = {}

                for candidate in pending_nodes:
                    score = 0

                    for blocks in netlist.nets.values():
                        if candidate in blocks:
                            already_ordered_neighbors = (
                                (blocks - {candidate}) & set(ordered)
                            )

                            score += len(already_ordered_neighbors)

                    scores[candidate] = score

                chosen = min(
                    pending_nodes,
                    key=lambda node_id: (-scores[node_id], node_id)
                )

                ordered.append(chosen)
                pending_nodes.remove(chosen)

    return ordered


class PlacementEnv:
    
    def __init__(self, netlist: Netlist, num_rows: int, num_cols: int) -> None:
        self.netlist: Netlist = netlist
        self.num_rows: int = num_rows
        self.num_cols: int = num_cols
        self.placement_order: list[str] = compute_placement_order(netlist)
        self.config: dict[str, tuple[int, int]] = {}
        self.current_idx: int = 0

        # Fixed node ordering + static graph topology (same for every step)
        self.node_order = list(netlist.nodes.keys())
        self.node_to_idx = {nid: i for i, nid in enumerate(self.node_order)}
        self.edge_index, self.edge_weight = self._build_edges()

    def reset(self) -> None:
        self.config = {}
        self.current_idx = 0


    def action_to_grid(self, action: int) -> tuple[int, int]:   # action in [0, num_rows*num_cols]
        row = action // self.num_cols
        col = action % self.num_cols
        return row, col
    

    def grid_to_action(self, row: int, col: int) -> int:
        return row * self.num_cols + col
    
    
    def grid_to_physical(self, row: int, col: int) -> tuple[float, float]:
        cell_width = self.netlist.canvas["width"] / self.num_cols
        cell_height = self.netlist.canvas["height"] / self.num_rows

        x = (col + 0.5) * cell_width
        y = (row + 0.5) * cell_height

        return x, y
    
    
    def get_rectangle(self, node_id: str, row: int, col: int) -> tuple[float, float, float, float]:
        x, y = self.grid_to_physical(row, col)

        width = self.netlist.nodes[node_id]["width"]
        height = self.netlist.nodes[node_id]["height"]

        left = x - width / 2
        right = x + width / 2
        bottom = y - height / 2
        top = y + height / 2

        return left, right, bottom, top
    

    def get_boundary_mask(self) -> list[bool]:
        current_node_id = self.placement_order[self.current_idx]

        mask = []

        for action in range(self.num_rows * self.num_cols):
            row, col = self.action_to_grid(action)

            left, right, bottom, top = self.get_rectangle(current_node_id, row, col)

            is_legal = (
                left >= 0
                and right <= self.netlist.canvas["width"]
                and bottom >= 0
                and top <= self.netlist.canvas["height"]
            )

            mask.append(is_legal)

        return mask
    

    def overlap_exists(self, node_id: str, row: int, col: int) -> bool: 
        left, right, bottom, top = self.get_rectangle(node_id, row, col)

        for placed_node_id, (placed_row, placed_col) in self.config.items():
            other_left, other_right, other_bottom, other_top = self.get_rectangle(placed_node_id, placed_row, placed_col)

            non_overlapping = (
                right <= other_left
                or left >= other_right
                or top <= other_bottom
                or bottom >= other_top
            )

            if not non_overlapping:
                return True

        return False
    

    def get_action_mask(self) -> list[bool]:
        current_node_id = self.placement_order[self.current_idx]

        boundary_mask = self.get_boundary_mask()
        action_mask = []

        for action in range(self.num_rows * self.num_cols):
            if not boundary_mask[action]:
                action_mask.append(False)
                continue

            row, col = self.action_to_grid(action)

            if self.overlap_exists(current_node_id, row, col):
                action_mask.append(False)
            else:
                action_mask.append(True)

        return action_mask
        
    
    def step(self, action: int) -> tuple[float, bool, bool]:

        if action < 0 or action >= self.num_rows * self.num_cols:
            raise ValueError(f"Action {action} is outside the action space")


        action_mask = self.get_action_mask()

        if not action_mask[action]:
            raise ValueError(f"Action {action} is illegal")

        # Get current node and action coordinates
        current_node_id = self.placement_order[self.current_idx]
        row, col = self.action_to_grid(action)

        # Place current node
        self.config[current_node_id] = (row, col)

        # Move to next node
        self.current_idx += 1

        # Successful completion -> returns reward
        if self.current_idx >= len(self.placement_order):
            reward = compute_hpwl(self.netlist, self.config, self.num_rows, self.num_cols)
            normalizer = (self.netlist.canvas["width"]+ self.netlist.canvas["height"])
            reward = -reward / normalizer

            return reward, True, False

        # Check whether next node has any legal actions
        next_action_mask = self.get_action_mask()

        # Have to replace this with something smarter
        if not any(next_action_mask):
            #return -2*(self.netlist.canvas["width"]+ self.netlist.canvas["height"]), True, True
            return (self.netlist.canvas["width"]+ self.netlist.canvas["height"])/100, True, True

        # Episode continues
        return 0.0, False, False
    
        # The return values represent reward, done, failed


    def get_observation(self) -> list[float]:
        observation = []

        for node_id in self.placement_order:
            if node_id in self.config:
                row, col = self.config[node_id]

                normalized_row = (row + 0.5) / self.num_rows
                normalized_col = (col + 0.5) / self.num_cols

                observation.extend([1.0, normalized_row, normalized_col])

            else:
                observation.extend([0.0, 0.0, 0.0])

        current_node_id = self.placement_order[self.current_idx]
        current_node_dims = self.netlist.nodes[current_node_id]      

        normalized_width = current_node_dims["width"] / self.netlist.canvas["width"]
        normalized_height = current_node_dims["height"] / self.netlist.canvas["height"]
        observation.extend([normalized_width, normalized_height])
        
        connectivity = []

        placed_nodes = set(self.config.keys())
 
        for node_id in self.placement_order:
            if node_id not in placed_nodes:
                connectivity.append(0.0)
                continue

            shared_net_count = 0

            for blocks in self.netlist.nets.values():
                if current_node_id in blocks and node_id in blocks:
                    shared_net_count += 1

            connectivity.append(float(shared_net_count))

        observation.extend(connectivity)

        return observation
    

    def get_metrics(self) -> dict:
        if self.current_idx != len(self.placement_order):
            raise RuntimeError("Placement is not complete.")

        hpwl = compute_hpwl(self.netlist, self.config, self.num_rows, self.num_cols)

        return {"hpwl": hpwl}


    def _build_edges(self):
        edges, weights = [], []

        for blocks in self.netlist.nets.values():
            block_list = list(blocks)
            for i in range(len(block_list)):
                for j in range(i + 1, len(block_list)):
                    u = self.node_to_idx[block_list[i]]
                    v = self.node_to_idx[block_list[j]]
                    if (u, v) in edges:
                        weights[edges.index((u, v))] += 1
                    else:
                        edges.append((u, v))
                        weights.append(1)

        edges_bi, weights_bi = [], []
        for (u, v), w in zip(edges, weights):
            edges_bi.append([u, v])
            edges_bi.append([v, u])
            weights_bi.append(w)
            weights_bi.append(w)

        edge_index = torch.tensor(edges_bi, dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(weights_bi, dtype=torch.float32)
        return edge_index, edge_weight


    def get_graph_observation(self) -> Data:
        node_features = []
        canvas_width = self.netlist.canvas["width"]
        canvas_height = self.netlist.canvas["height"]

        for node_id in self.node_order:
            width = self.netlist.nodes[node_id]["width"] / canvas_width
            height = self.netlist.nodes[node_id]["height"] / canvas_height

            if node_id in self.config:
                row, col = self.config[node_id]
                x_grid = (col + 0.5) / self.num_cols
                y_grid = (row + 0.5) / self.num_rows
                is_placed = 1.0
            else:
                x_grid, y_grid, is_placed = 0.0, 0.0, 0.0

            node_features.append([width, height, x_grid, y_grid, is_placed])

        x = torch.tensor(node_features, dtype=torch.float32)

        current_node_id = self.placement_order[self.current_idx]
        current_node_idx = torch.tensor([self.node_to_idx[current_node_id]], dtype=torch.long)

        data = Data(x=x, edge_index=self.edge_index, edge_weight=self.edge_weight)
        data.current_node_idx = current_node_idx  # local index into this graph's x

        return data