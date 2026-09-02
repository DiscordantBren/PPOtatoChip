import torch
from torch.optim import Adam
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

from .netlist import Netlist
from .environment import PlacementEnv
from .models import PolicyValueNetwork, RewardPredictor, GraphPolicyValueNetwork 
from .rollout import run_episode
from .ppo import ppo_update
from .experiment import Experiment
from .reward_dataset import RewardDataSet
from .graph_rollout import run_graph_episode
from .graph_ppo import graph_ppo_update


def train_VanillaPVN(
        netlist_path: str,
        num_rows: int,
        num_cols: int,
        hidden_dim: int,
        num_hidden: int,
        gamma: float=0.99,
        clip_epsilon: float=0.2,
        value_loss_coef: float=0.5,
        entropy_coef: float=0.01,
        num_iterations: int=10,
        lr: float=3e-4,
        progress_callback=None,
        stop_event=None,
        ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netlist = Netlist(netlist_path)

    env = PlacementEnv(netlist=netlist, num_rows=num_rows, num_cols=num_cols)

    env.reset()

    input_dim = len(env.get_observation())
    num_actions = env.num_rows * env.num_cols

    model = PolicyValueNetwork(input_dim=input_dim, hidden_dim=hidden_dim, num_hidden = num_hidden, output_dim=num_actions).to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    experiment = Experiment(
                    config={
                        "iterations": num_iterations,
                        "learning_rate": lr,
                        "gamma": gamma,
                        "clip_epsilon": clip_epsilon,
                        "value_loss_coef": value_loss_coef,
                        "entropy_coef": entropy_coef,
                        "hidden_dim": hidden_dim,
                        "num_hidden": num_hidden,
                        "grid_rows": env.num_rows,
                        "grid_cols": env.num_cols,
                        "netlist_path": netlist_path,
                    },
                    tag="vanilla_pvn",
                )

    for iteration in range(num_iterations):

        if stop_event and stop_event.is_set():
            print("Training stopped by user.")
            break

        rollout = run_episode(env=env, model=model, device=device)

        if rollout.failed:
            print("Episode failed.")
        else:
            experiment.append_sample(
                netlist_name=netlist.name,
                placement=env.config,
                metrics=env.get_metrics(),
                failed=False
            )

        metrics = ppo_update(
            rollout=rollout,
            model=model,
            optimizer=optimizer,
            gamma=gamma,
            clip_epsilon=clip_epsilon,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            device=device
        )

        total_reward = sum(rollout.rewards)

        print(f"\nIteration {iteration + 1}")

        for epoch in range(len(metrics["loss"])):
            print(
                f"  Epoch {epoch + 1} | "
                f"Loss {metrics['loss'][epoch]:8.4f} | "
                f"Policy_Loss {metrics['policy_loss'][epoch]:8.4f} | "
                f"Value_Loss {metrics['value_loss'][epoch]:8.4f} | "
                f"Entropy {metrics['entropy'][epoch]:8.4f}"
            )

        print(
            f"Reward {total_reward:8.3f} | "
            f"Steps {len(rollout.actions):3d}"
        )

        hpwl = env.get_metrics().get("hpwl", None) if not rollout.failed else None

        iteration_metrics = {
            "iteration": iteration,
            "reward": total_reward,
            "steps": len(rollout.actions),
            "failed": rollout.failed,
            "hpwl": hpwl,
            "loss_mean": sum(metrics["loss"]) / len(metrics["loss"]),
            "policy_loss_mean": sum(metrics["policy_loss"]) / len(metrics["policy_loss"]),
            "value_loss_mean": sum(metrics["value_loss"]) / len(metrics["value_loss"]),
            "entropy_mean": sum(metrics["entropy"]) / len(metrics["entropy"]),
        }

        experiment.append_metrics(iteration_metrics)

        if progress_callback:
            progress_callback(iteration_metrics)

    # saves the parameters for the vanilla ppo after training
    experiment.save_model(model, "ppo_initial.pt")

    return experiment



def train_RewardPredictor(
        placements_path: str,
        netlists: dict[str, Netlist],
        num_rows: int,
        num_cols: int,
        hidden_channels_e: int = 128,
        num_layers_e: int = 3,
        hidden_channels_r: int = 128,
        num_layers_r: int = 3,
        batch_size: int = 32,
        lr: float = 3e-4,
        num_epochs: int = 50,
        progress_callback=None,
        stop_event=None,
        ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = RewardDataSet(
                placements_path=placements_path,
                netlists=netlists,
                num_rows=num_rows,
                num_cols=num_cols,
            )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    sample = dataset.get(0)
    assert sample.x is not None
    in_channels = sample.x.shape[1]

    model = RewardPredictor(
        in_channels=in_channels,
        hidden_channels_e=hidden_channels_e,
        num_layers_e=num_layers_e,
        hidden_channels_r=hidden_channels_r,
        num_layers_r=num_layers_r,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=lr)

    experiment = Experiment(
                    config={
                        "epochs": num_epochs,
                        "learning_rate": lr,
                        "batch_size": batch_size,
                        "hidden_channels_e": hidden_channels_e,
                        "num_layers_e": num_layers_e,
                        "hidden_channels_r": hidden_channels_r,
                        "num_layers_r": num_layers_r,
                        "in_channels": in_channels,
                    },
                    tag="reward_predictor",
                )

    for epoch in range(num_epochs):

        if stop_event and stop_event.is_set():
            print("Training stopped by user.")
            break

        model.train()
        total_loss = 0.0

        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            preds = model(
                batch.x,
                batch.edge_index,
                batch.edge_weight,
                batch.batch,
            )

            loss = F.mse_loss(preds, batch.y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.num_graphs

        avg_loss = total_loss / len(dataset)

        print(f"Epoch {epoch + 1:3d} | MSE Loss {avg_loss:8.4f}")

        if progress_callback:
            progress_callback({"epoch": epoch, "mse_loss": avg_loss})

    # saves the parameters for the reward predictor (encoder used for later pretraining transfer)
    experiment.save_model(model, "reward_predictor.pt")
    experiment.save_model(model.encoder, "encoder.pt")

    return model, experiment





def train_GraphPPO(
        netlist_path: str,
        num_rows: int,
        num_cols: int,
        hidden_channels_e: int = 128,
        num_layers_e: int = 3,
        hidden_dim: int = 128,
        num_hidden: int = 2,
        pretrained_encoder_path: str | None = None,
        freeze_encoder: bool = False,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        num_iterations: int = 10,
        lr: float = 3e-4,
        progress_callback=None,
        stop_event=None,
        ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netlist = Netlist(netlist_path)

    env = PlacementEnv(netlist=netlist, num_rows=num_rows, num_cols=num_cols)

    env.reset()

    graph_obs = env.get_graph_observation()
    assert graph_obs.x is not None
    in_channels = graph_obs.x.shape[1]
    num_actions = env.num_rows * env.num_cols

    model = GraphPolicyValueNetwork(
        output_dim=num_actions,
        in_channels=in_channels,
        hidden_channels_e=hidden_channels_e,
        num_layers_e=num_layers_e,
        hidden_dim=hidden_dim,
        num_hidden=num_hidden,
    ).to(device)

    if pretrained_encoder_path is not None:
        model.encoder.load_state_dict(torch.load(pretrained_encoder_path, map_location=device))
        print(f"Loaded pretrained encoder from {pretrained_encoder_path}")

        if freeze_encoder:
            for param in model.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen.")

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    experiment = Experiment(
                    config={
                        "iterations": num_iterations,
                        "learning_rate": lr,
                        "gamma": gamma,
                        "clip_epsilon": clip_epsilon,
                        "value_loss_coef": value_loss_coef,
                        "entropy_coef": entropy_coef,
                        "hidden_channels_e": hidden_channels_e,
                        "num_layers_e": num_layers_e,
                        "hidden_dim": hidden_dim,
                        "num_hidden": num_hidden,
                        "grid_rows": env.num_rows,
                        "grid_cols": env.num_cols,
                        "pretrained_encoder_path": pretrained_encoder_path,
                        "freeze_encoder": freeze_encoder,
                        "netlist_path": netlist_path,
                    },
                    tag="graph_ppo",
                )

    for iteration in range(num_iterations):

        if stop_event and stop_event.is_set():
            print("Training stopped by user.")
            break

        rollout = run_graph_episode(env=env, model=model, device=device)

        if rollout.failed:
            print("Episode failed.")
        else:
            experiment.append_sample(
                netlist_name=netlist.name,
                placement=env.config,
                metrics=env.get_metrics(),
                failed=False,
            )

        metrics = graph_ppo_update(
            rollout=rollout,
            model=model,
            optimizer=optimizer,
            gamma=gamma,
            clip_epsilon=clip_epsilon,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            device=device
        )

        total_reward = sum(rollout.rewards)

        print(f"\nIteration {iteration + 1}")

        for epoch in range(len(metrics["loss"])):
            print(
                f"  Epoch {epoch + 1} | "
                f"Loss {metrics['loss'][epoch]:8.4f} | "
                f"Policy_Loss {metrics['policy_loss'][epoch]:8.4f} | "
                f"Value_Loss {metrics['value_loss'][epoch]:8.4f} | "
                f"Entropy {metrics['entropy'][epoch]:8.4f}"
            )

        print(
            f"Reward {total_reward:8.3f} | "
            f"Steps {len(rollout.actions):3d}"
        )

        hpwl = env.get_metrics().get("hpwl", None) if not rollout.failed else None

        iteration_metrics = {
            "iteration": iteration,
            "reward": total_reward,
            "steps": len(rollout.actions),
            "failed": rollout.failed,
            "hpwl": hpwl,
            "loss_mean": sum(metrics["loss"]) / len(metrics["loss"]),
            "policy_loss_mean": sum(metrics["policy_loss"]) / len(metrics["policy_loss"]),
            "value_loss_mean": sum(metrics["value_loss"]) / len(metrics["value_loss"]),
            "entropy_mean": sum(metrics["entropy"]) / len(metrics["entropy"]),
        }

        experiment.append_metrics(iteration_metrics)

        if progress_callback:
            progress_callback(iteration_metrics)

    experiment.save_model(model, "graph_ppo_final.pt")

    return model, experiment