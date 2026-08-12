import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .models import PolicyValueNetwork
from .rollout import Rollout


def compute_returns(rewards: list[float], gamma: float) -> torch.Tensor:

    returns = []
    running_return = 0.0

    for reward in reversed(rewards):

        running_return = reward + gamma * running_return
        returns.insert(0, running_return)

    return torch.tensor(returns, dtype=torch.float32)


def compute_advantages(returns: torch.Tensor, values: torch.Tensor) -> torch.Tensor:

    return returns - values


def ppo_update(
    rollout: Rollout,
    model: PolicyValueNetwork,
    optimizer: torch.optim.Optimizer,
    gamma: float,
    clip_epsilon: float,
    value_loss_coef: float,
    entropy_coef: float,
    device: torch.device,
    ppo_epochs=4
):

    returns = compute_returns(rollout.rewards, gamma).to(device)
    observations = torch.stack(rollout.observations)        # torch.stack converts the list of tensors into a tensor
    action_masks = torch.stack(rollout.action_masks)
    actions = torch.stack(rollout.actions)
    old_log_probs = torch.stack(rollout.log_probs)
    old_values = torch.stack(rollout.values)

    advantages = compute_advantages(returns, old_values)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # initializing the loss history for all epochs
    history = {"loss": [], "policy_loss": [], "value_loss": [], "entropy": []}

    # Run the model again multiple times for sample efficiency
    for epoch in range(ppo_epochs):

        logits, values = model(observations)

        masked_logits = logits.masked_fill(~action_masks, -torch.inf)
        distribution = Categorical(logits=masked_logits)

        new_log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy().mean()

        ratio = torch.exp(new_log_probs - old_log_probs)

        clipped_ratio = torch.clamp(
            ratio,
            1.0 - clip_epsilon,
            1.0 + clip_epsilon,
        )

        surrogate1 = ratio * advantages
        surrogate2 = clipped_ratio * advantages

        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        value_loss = F.mse_loss(values.squeeze(-1), returns)

        loss = (
            policy_loss
            + value_loss_coef * value_loss
            - entropy_coef * entropy
        )

        history["loss"].append(loss.item())
        history["policy_loss"].append(policy_loss.item())
        history["value_loss"].append(value_loss.item())
        history["entropy"].append(entropy.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return history