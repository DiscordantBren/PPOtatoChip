"""
Live training monitor screen. Runs training in a background thread
with safe UI updates via call_from_thread.
"""

import threading
import time
from pathlib import Path

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static, Label, ProgressBar, RichLog, Button
from textual.containers import Horizontal, Vertical
from textual import events

from .._utils import footer_bar
from ...training import train_VanillaPVN, train_RewardPredictor, train_GraphPPO
from ...netlist import Netlist


class TrainProgressScreen(Screen):
    """Shows live metrics, progress bar, and console log during training."""

    def __init__(self, config: dict, experiment_label: str = "Training",
                 run_index: int = 0, total_runs: int = 1):
        super().__init__()
        self.config = config
        self.experiment_label = experiment_label
        self.run_index = run_index
        self.total_runs = total_runs
        self._stop_event = threading.Event()
        self._thread = None
        self._result = None
        self._error = None

    def compose(self) -> ComposeResult:
        label = self.experiment_label
        if self.total_runs > 1:
            label += f" (Run {self.run_index + 1}/{self.total_runs})"
        yield Static(label, id="wizard-header")

        with Vertical(id="progress-container"):
            with Vertical(id="progress-header"):
                yield Static("Initializing...", id="status-label")
                with Horizontal(id="progress-bar"):
                    yield ProgressBar(total=100, show_eta=False, id="pbar")

            with Horizontal(classes="metric-grid"):
                yield Vertical(Static("Reward"), Static("--", id="m_reward"), classes="metric-card")
                yield Vertical(Static("HPWL"), Static("--", id="m_hpwl"), classes="metric-card")
                yield Vertical(Static("Steps"), Static("--", id="m_steps"), classes="metric-card")
                yield Vertical(Static("Loss"), Static("--", id="m_loss"), classes="metric-card")
                yield Vertical(Static("Entropy"), Static("--", id="m_entropy"), classes="metric-card")

            with Vertical(id="log-container"):
                yield Label("Console Output")
                yield RichLog(id="log-view", highlight=True, max_lines=500)

            with Horizontal(id="action-bar"):
                yield Button("Cancel Training", id="cancel-btn", classes="action-btn", variant="error")

        yield footer_bar("c Cancel  Esc Back")

    def on_mount(self) -> None:
        self.query_one("#pbar").update(total=100)
        self._start_training()

    def _start_training(self) -> None:
        self._thread = threading.Thread(target=self._run_training, daemon=True)
        self._thread.start()

    def _run_training(self) -> None:
        """Run in a background thread. UI updates via call_from_thread."""
        try:
            config = self.config
            mode = config["mode"]

            def progress_callback(m: dict) -> None:
                if self._stop_event.is_set():
                    return
                self.app.call_from_thread(self._on_progress, m)

            if mode == "vanilla_pvn":
                result = train_VanillaPVN(
                    netlist_path=config["netlist_path"],
                    num_rows=config["num_rows"], num_cols=config["num_cols"],
                    hidden_dim=config["hidden_dim"], num_hidden=config["num_hidden"],
                    gamma=config["gamma"], clip_epsilon=config["clip_epsilon"],
                    value_loss_coef=config["value_loss_coef"], entropy_coef=config["entropy_coef"],
                    num_iterations=config["num_iterations"], lr=config["lr"],
                    progress_callback=progress_callback,
                    stop_event=self._stop_event,
                )
                self._result = ("vanilla_pvn", result)

            elif mode == "reward_predictor":
                np = config["netlist_path"]
                netlists = {Path(np).stem: Netlist(np)}
                model, exp = train_RewardPredictor(
                    placements_path=config.get("placements_path", ""),
                    netlists=netlists,
                    num_rows=config["num_rows"], num_cols=config["num_cols"],
                    hidden_channels_e=config["hidden_channels_e"], num_layers_e=config["num_layers_e"],
                    hidden_channels_r=config["hidden_channels_r"], num_layers_r=config["num_layers_r"],
                    batch_size=config["batch_size"], lr=config["lr"],
                    num_epochs=config["num_epochs"],
                    progress_callback=progress_callback,
                    stop_event=self._stop_event,
                )
                self._result = ("reward_predictor", model, exp)

            elif mode == "graph_ppo":
                model, exp = train_GraphPPO(
                    netlist_path=config["netlist_path"],
                    num_rows=config["num_rows"], num_cols=config["num_cols"],
                    hidden_channels_e=config["hidden_channels_e"], num_layers_e=config["num_layers_e"],
                    hidden_dim=config["hidden_dim"], num_hidden=config["num_hidden"],
                    pretrained_encoder_path=config.get("pretrained_encoder_path"),
                    freeze_encoder=config.get("freeze_encoder", False),
                    gamma=config["gamma"], clip_epsilon=config["clip_epsilon"],
                    value_loss_coef=config["value_loss_coef"], entropy_coef=config["entropy_coef"],
                    num_iterations=config["num_iterations"], lr=config["lr"],
                    progress_callback=progress_callback,
                    stop_event=self._stop_event,
                )
                self._result = ("graph_ppo", model, exp)

            elif mode == "full_pipeline":
                self.app.call_from_thread(self._log, "=== Stage 1: Vanilla PVN ===")
                exp_vp = train_VanillaPVN(
                    netlist_path=config["netlist_path"],
                    num_rows=config["num_rows"], num_cols=config["num_cols"],
                    hidden_dim=config["hidden_dim"], num_hidden=config["num_hidden"],
                    gamma=config["gamma"], clip_epsilon=config["clip_epsilon"],
                    value_loss_coef=config["value_loss_coef"], entropy_coef=config["entropy_coef"],
                    num_iterations=config["vanilla_iterations"], lr=config["lr"],
                    progress_callback=progress_callback,
                    stop_event=self._stop_event,
                )
                if self._stop_event.is_set():
                    return

                self.app.call_from_thread(self._log, "=== Stage 2: Reward Predictor ===")
                np = config["netlist_path"]
                netlists = {Path(np).stem: Netlist(np)}
                rp = config.get("reward_predictor", {})
                _, exp_rp = train_RewardPredictor(
                    placements_path=str(exp_vp.path / "placements.jsonl"),
                    netlists=netlists, num_rows=config["num_rows"], num_cols=config["num_cols"],
                    hidden_channels_e=config["hidden_channels_e"], num_layers_e=config["num_layers_e"],
                    hidden_channels_r=rp.get("hidden_channels_r", 128),
                    num_layers_r=rp.get("num_layers_r", 3),
                    batch_size=rp.get("batch_size", 32), num_epochs=rp.get("num_epochs", 100),
                    lr=config["lr"], progress_callback=progress_callback,
                    stop_event=self._stop_event,
                )
                if self._stop_event.is_set():
                    return

                self.app.call_from_thread(self._log, "=== Stage 3: Graph PPO ===")
                pp = config.get("graph_ppo", {})
                encoder_path = exp_rp.path / "encoder.pt"
                model, exp = train_GraphPPO(
                    netlist_path=config["netlist_path"],
                    num_rows=config["num_rows"], num_cols=config["num_cols"],
                    hidden_channels_e=config["hidden_channels_e"], num_layers_e=config["num_layers_e"],
                    hidden_dim=config["hidden_dim"], num_hidden=config["num_hidden"],
                    pretrained_encoder_path=str(encoder_path),
                    freeze_encoder=pp.get("freeze_encoder", False),
                    gamma=config["gamma"], clip_epsilon=config["clip_epsilon"],
                    value_loss_coef=config["value_loss_coef"], entropy_coef=config["entropy_coef"],
                    num_iterations=pp.get("num_iterations", 100), lr=config["lr"],
                    progress_callback=progress_callback,
                    stop_event=self._stop_event,
                )
                self._result = ("full_pipeline", model, exp)

            if not self._stop_event.is_set():
                self.app.call_from_thread(self._on_done)

        except Exception as e:
            import traceback
            self._error = f"{type(e).__name__}: {e}"
            self.app.call_from_thread(self._on_error, traceback.format_exc())

    def _on_progress(self, metrics: dict) -> None:
        total = self.config.get("num_iterations", self.config.get("num_epochs", 100))
        # For full pipeline, use the correct stage-specific total
        stage = metrics.get("stage", "")
        if "Vanilla" in stage:
            total = self.config.get("vanilla_iterations", total)
        elif "Graph" in stage:
            pp = self.config.get("graph_ppo", {})
            total = pp.get("num_iterations", total)
        elif "Reward" in stage:
            rp = self.config.get("reward_predictor", {})
            total = rp.get("num_epochs", total)

        iteration = metrics.get("iteration", metrics.get("epoch", 0))
        pct = min(100.0, (iteration + 1) / total * 100.0)
        try:
            self.query_one("#pbar").progress = pct
            self.query_one("#status-label").update(f"{stage} {iteration + 1}/{total} ({pct:.0f}%)")

            for key, wid in [("reward", "m_reward"), ("hpwl", "m_hpwl"), ("steps", "m_steps"),
                             ("loss_mean", "m_loss"), ("entropy_mean", "m_entropy")]:
                val = metrics.get(key)
                if val is not None:
                    try:
                        self.query_one(f"#{wid}").update(f"{val:.4f}")
                    except Exception:
                        pass

            self._log(f"Iter {iteration + 1}: reward={metrics.get('reward', '?'):.3f} hpwl={metrics.get('hpwl', '?'):.2f}")
        except Exception:
            pass

    def _log(self, text: str) -> None:
        try:
            self.query_one("#log-view").write(text)
        except Exception:
            pass

    def _on_done(self) -> None:
        self._log("\n[green]Training complete![/green]")
        time.sleep(0.3)
        self.app.pop_screen()
        from .train_complete import TrainCompleteScreen
        self.app.push_screen(TrainCompleteScreen(result=self._result, config=self.config))

    def _on_error(self, tb: str) -> None:
        self._log(f"[red]ERROR: {self._error}[/red]")
        self._log(tb)
        self.query_one("#status-label").update("Training Failed")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel-btn":
            self.action_cancel()

    def on_key(self, event: events.Key) -> None:
        if event.key == "c":
            self.action_cancel()
            event.stop()
        elif event.key == "escape":
            self.app.pop_screen()
            event.stop()

    def action_cancel(self) -> None:
        self._stop_event.set()
        self._log("[yellow]Cancelling...[/yellow]")
        try:
            self.query_one("#status-label").update("Cancelling...")
            self.query_one("#cancel-btn").disabled = True
        except Exception:
            pass