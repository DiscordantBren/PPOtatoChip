from dataclasses import dataclass

import torch
from torch.distributions import Categorical
from torch_geometric.data import Data

from .environment import PlacementEnv
from .models import GraphPolicyValueNetwork


@dataclass
class GraphRollout:
    observations: list[Data]
    action_masks: list[torch.Tensor]
    actions: list[torch.Tensor]
    log_probs: list[torch.Tensor]
    values: list[torch.Tensor]
    rewards: list[float]
    dones: list[bool]
    failed: bool


def run_graph_episode(env: PlacementEnv, model: GraphPolicyValueNetwork, device: torch.device) -> GraphRollout:

    observations = []
    action_masks = []
    actions = []
    log_probs = []
    values = []
    rewards = []
    dones = []

    env.reset()

    done = False
    failed = False

    while not done:

        graph_obs = env.get_graph_observation().to(device)

        action_mask = torch.as_tensor(env.get_action_mask(), dtype=torch.bool, device=device)

        with torch.no_grad():
            logits, value = model(
                graph_obs.x,
                graph_obs.edge_index,
                graph_obs.edge_weight,
                graph_obs.current_node_idx,
            )

            logits = logits.squeeze(0)
            value = value.squeeze(0)

            masked_logits = logits.masked_fill(~action_mask, -torch.inf)
            distribution = Categorical(logits=masked_logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)

        reward, done, failed = env.step(int(action.item()))

        observations.append(graph_obs)
        action_masks.append(action_mask)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value.squeeze(-1))

        rewards.append(reward)
        dones.append(done)

    return GraphRollout(
        observations=observations,
        action_masks=action_masks,
        actions=actions,
        log_probs=log_probs,
        values=values,
        rewards=rewards,
        dones=dones,
        failed=failed,
    )