from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from .environment import PlacementEnv
from .models import PolicyValueNetwork


@dataclass
class Rollout:
    observations: list[torch.Tensor]
    action_masks: list[torch.Tensor]
    actions: list[torch.Tensor]
    log_probs: list[torch.Tensor]
    values: list[torch.Tensor]
    rewards: list[float]
    dones: list[bool]
    failed: bool


def run_episode(env: PlacementEnv, model: PolicyValueNetwork, device: torch.device) -> Rollout:

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

        # as_tensor is more memory efficient
        observation = torch.as_tensor(env.get_observation(), dtype=torch.float32, device=device)

        action_mask = torch.as_tensor(env.get_action_mask(), dtype=torch.bool, device=device)

        # Forward pass with the quantities detached from computation graph
        with torch.no_grad():
            logits, value = model(observation)

            masked_logits = logits.masked_fill(~action_mask, -torch.inf)
            distribution = Categorical(logits=masked_logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)

        reward, done, failed = env.step(int(action.item()))

        observations.append(observation)
        action_masks.append(action_mask)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value.squeeze(-1))

        rewards.append(reward)
        dones.append(done)

    return Rollout(
        observations=observations,
        action_masks=action_masks,
        actions=actions,
        log_probs=log_probs,
        values=values,
        rewards=rewards,
        dones=dones,
        failed=failed,
    )