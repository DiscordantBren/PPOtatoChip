from typing import Any

import torch
import torch.nn as nn
from torch_geometric.nn import GraphConv, global_mean_pool

class PolicyValueNetwork(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, num_hidden: int, output_dim: int) -> None:

        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim)]
        layers += [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden - 1)]
        self.fc_layers = nn.ModuleList(layers)

        self.activation = nn.ReLU()
        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)


    def forward(self, x):
        for layer in self.fc_layers:
            x = self.activation(layer(x))

        logits = self.policy_head(x)
        value = self.value_head(x)
        
        return logits, value


class GraphConvEncoder(nn.Module):

    def __init__(self, in_channels: int = 5, hidden_channels: int = 128, num_layers: int = 3) -> None:
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(in_channels, hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(GraphConv(hidden_channels, hidden_channels))

        self.activation = nn.ReLU()

    def forward(self, x, edge_index, edge_weight):

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_weight)

            if i < len(self.convs) - 1:
                x = self.activation(x)

        return x


class RewardPredictor(nn.Module):

    def __init__(
            self,
            in_channels: int = 5,
            hidden_channels_e: int = 128,
            num_layers_e: int = 3,
            hidden_channels_r: int = 128,
            num_layers_r: int = 3
            ) -> None:
        
        super().__init__()

        self.encoder = GraphConvEncoder(in_channels, hidden_channels_e, num_layers_e)
        self.reward_layers = nn.ModuleList([nn.Linear(hidden_channels_e, hidden_channels_r), nn.ReLU()])

        for i in range(num_layers_r - 2):
            self.reward_layers.append(nn.Linear(hidden_channels_r, hidden_channels_r))
            self.reward_layers.append(nn.ReLU())

        self.reward_layers.append(nn.Linear(hidden_channels_r, 1))


    def forward(self, x, edge_index, edge_weight, batch=None):

        node_embeddings = self.encoder(x, edge_index, edge_weight)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        graph_embedding = global_mean_pool(node_embeddings, batch)
        y = graph_embedding

        for layer in self.reward_layers:
            y = layer(y)

        return y.squeeze(-1)


class GraphPolicyValueNetwork(nn.Module):

    def __init__(
            self,
            output_dim: int,
            in_channels: int = 5,
            hidden_channels_e: int = 128,
            num_layers_e: int = 3,
            hidden_dim: int = 128,
            num_hidden: int = 3,
            ):
        super().__init__()

        self.encoder = GraphConvEncoder(in_channels, hidden_channels_e, num_layers_e)
        self.embed_norm = nn.LayerNorm(hidden_channels_e)

        combined_dim = hidden_channels_e * 2    # current-node embedding + pooled graph embedding

        layers = [nn.Linear(combined_dim, hidden_dim)]
        layers += [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden - 1)]
        self.fc_layers = nn.ModuleList(layers)
        self.activation = nn.ReLU()

        self.policy_head = nn.Linear(hidden_dim, output_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_weight, current_node_idx, batch=None):

        node_embeddings = self.encoder(x, edge_index, edge_weight)      # [total_nodes, hidden_channels_e]
        node_embeddings = self.embed_norm(node_embeddings)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        graph_embedding = global_mean_pool(node_embeddings, batch)      # [num_graphs, hidden_channels_e]
        current_embedding = node_embeddings[current_node_idx]           # [num_graphs, hidden_channels_e]

        h = torch.cat([current_embedding, graph_embedding], dim=-1)     # [num_graphs, 2*hidden_channels_e]

        for layer in self.fc_layers:
            h = self.activation(layer(h))

        logits = self.policy_head(h)
        value = self.value_head(h)

        return logits, value
