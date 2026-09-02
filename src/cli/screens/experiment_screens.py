# Experiment wizard - setup, progress, results.

import threading
from pathlib import Path

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static, Button, ProgressBar, RichLog
from textual.containers import Horizontal, Vertical
from textual import events

from .._utils import available_netlists, footer_bar
from ..experiments.recipes import PREDEFINED_RECIPES
from ..experiments.runner import run_experiment


class ExperimentSetupScreen(Screen):

    def __init__(self):
        super().__init__()
        self._phase = "type"
        self._sel = 0
        self._mode = "recipe"
        self._recipe_idx = 0
        self._sweep_param = "lr"
        self._sweep_values = [64, 128, 256]
        self._value_spec = "explicit"
        self._netlist_idx = 0
        self._param_list = [
            ("lr", "Learning Rate"), ("gamma", "Gamma"), ("clip_epsilon", "Clip Epsilon"),
            ("value_loss_coef", "Value Loss Coef"), ("entropy_coef", "Entropy Coef"),
            ("hidden_dim", "MLP Dim"), ("num_hidden", "MLP Layers"),
            ("hidden_channels_e", "GNN Ch"), ("num_layers_e", "GNN Layers"),
            ("num_iterations", "Iterations"), ("num_rows", "Rows"), ("num_cols", "Cols"),
            ("batch_size", "Batch"), ("num_epochs", "Epochs"),
            ("freeze_encoder", "Freeze Encoder"),
        ]
        self._num_runs = 3
        self._delete_artifacts = False
        self._widgets = []
        self._edit_mode = False
        self._edit_buffer = ""

    def compose(self) -> ComposeResult:
        yield Static("Run Experiment")
        yield Static("")
        with Vertical(id="wizard-container"):
            for _ in range(20):
                w = Static("")
                self._widgets.append(w)
                yield w
        yield footer_bar("j Down  k Up  Enter Select  n Next  Esc Back")

    def on_mount(self) -> None:
        self._rebuild()

    def _rebuild(self):
        lines = []
        nets = available_netlists() or ["(no netlists)"]

        if self._phase == "type":
            lines.append(("header", "Experiment type:"))
            items = ["  Predefined recipe", "  Custom sweep"]
            for i, item in enumerate(items):
                p = ">" if i == self._sel else " "
                t = "option" if i == self._sel else "text"
                lines.append((t, f"  {p} {item}"))

        elif self._phase == "recipe":
            lines.append(("header", "Recipe:"))
            for i, r in enumerate(PREDEFINED_RECIPES):
                p = ">" if i == self._sel else " "
                t = "option" if i == self._sel else "text"
                lines.append((t, f"  {p} {r['name']}"))

        elif self._phase == "param":
            lines.append(("header", "Parameter to sweep:"))
            for i, (key, label) in enumerate(self._param_list):
                p = ">" if i == self._sel else " "
                t = "option" if i == self._sel else "text"
                lines.append((t, f"  {p} {label}"))

        elif self._phase == "values":
            lines.append(("header", "Value specification:"))
            items = ["  Explicit values (comma-separated)", "  Range with step"]
            for i, item in enumerate(items):
                p = ">" if i == self._sel else " "
                t = "option" if i == self._sel else "text"
                lines.append((t, f"  {p} {item}"))

        elif self._phase == "values_explicit":
            if self._edit_mode:
                lines.append(("header", "Enter values (comma-separated):"))
                lines.append(("text", f"  [{self._edit_buffer}]"))
            else:
                lines.append(("header", f"Values: {self._sweep_values}"))
                lines.append(("option", "  Press Enter to edit, n for next"))

        elif self._phase == "values_range":
            if self._edit_mode:
                lines.append(("header", "Enter range (from,to,step):"))
                lines.append(("text", f"  [{self._edit_buffer}]"))
            else:
                lines.append(("header", f"Values: {self._sweep_values}"))
                lines.append(("option", "  Press Enter to edit, n for next"))

        elif self._phase == "num_runs":
            if self._edit_mode:
                lines.append(("header", "Runs per value:"))
                lines.append(("text", f"  [{self._edit_buffer}]"))
            else:
                lines.append(("header", "Runs per value:"))
                lines.append(("option", f"  {self._num_runs}"))

        elif self._phase == "netlist":
            lines.append(("header", "Netlist:"))
            for i, n in enumerate(nets):
                p = ">" if i == self._sel else " "
                t = "option" if i == self._sel else "text"
                lines.append((t, f"  {p} {n}"))

        elif self._phase == "delete":
            lines.append(("header", "Delete artifacts after report?"))
            items = ["  No", "  Yes"]
            for i, item in enumerate(items):
                p = ">" if i == self._sel else " "
                t = "option" if i == self._sel else "text"
                lines.append((t, f"  {p} {item}"))

        elif self._phase == "summary":
            net_name = nets[self._netlist_idx] if self._netlist_idx < len(nets) else "?"
            lines.append(("header", "Experiment Summary:"))
            lines.append(("text", f"  Mode: {self._mode}"))
            lines.append(("text", f"  Netlist: {net_name}"))
            lines.append(("text", f"  Sweep param: {self._sweep_param}"))
            lines.append(("text", f"  Values: {self._sweep_values}"))
            lines.append(("text", f"  Runs per value: {self._num_runs}"))

        for i, w in enumerate(self._widgets):
            if i < len(lines):
                typ, text = lines[i]
                w.update(text)
                if typ == "option":
                    w.styles.background = "#ffffff"
                    w.styles.color = "#000000"
                elif typ == "header":
                    w.styles.background = "#000000"
                    w.styles.color = "#ffffff"
                else:
                    w.styles.background = "#000000"
                    w.styles.color = "#ffffff"
            else:
                w.update("")

    def _max_sel(self) -> int:
        if self._phase == "type":
            return 2
        elif self._phase == "recipe":
            return len(PREDEFINED_RECIPES)
        elif self._phase == "param":
            return len(self._param_list)
        elif self._phase == "values":
            return 2
        elif self._phase in ("values_explicit", "values_range", "num_runs"):
            return 1
        elif self._phase == "delete":
            return 2
        elif self._phase == "netlist":
            nets = available_netlists() or ["(no netlists)"]
            return len(nets)
        return 1

    def on_key(self, event: events.Key) -> None:
        if self._edit_mode:
            if event.key == "enter":
                self._edit_mode = False
                if self._phase in ("values_explicit", "values_range"):
                    self._sweep_values = self._parse_values(self._edit_buffer)
                elif self._phase == "num_runs":
                    try:
                        self._num_runs = max(1, int(self._edit_buffer))
                    except ValueError:
                        self._num_runs = 3
                self._rebuild()
                event.stop()
                return
            elif event.key == "escape":
                self._edit_mode = False
                self._rebuild()
                event.stop()
                return
            elif event.key == "backspace":
                self._edit_buffer = self._edit_buffer[:-1]
                self._rebuild()
                event.stop()
                return
            elif event.key == "n":
                if self._phase in ("values_explicit", "values_range"):
                    self._sweep_values = self._parse_values(self._edit_buffer)
                elif self._phase == "num_runs":
                    try:
                        self._num_runs = max(1, int(self._edit_buffer))
                    except ValueError:
                        self._num_runs = 3
                self._edit_mode = False
                self._rebuild()
            elif event.key == "comma" or event.key == ",":
                self._edit_buffer += ","
                self._rebuild()
                event.stop()
                return
            elif len(event.key) == 1:
                self._edit_buffer += event.key
                self._rebuild()
                event.stop()
                return
            else:
                return

        max_sel = self._max_sel()
        if event.key == "j":
            self._sel = (self._sel + 1) % max_sel
            self._rebuild()
            event.stop()
        elif event.key == "k":
            self._sel = (self._sel - 1) % max_sel
            self._rebuild()
            event.stop()
        elif event.key == "escape":
            self._go_back()
            event.stop()
        elif event.key == "enter":
            self._handle_enter()
            event.stop()
        elif event.key == "n":
            self._handle_next()
            event.stop()

    def _go_back(self):
        if self._phase == "type":
            self.app.pop_screen()
        elif self._phase in ("recipe", "param"):
            self._phase = "type"
            self._sel = 0 if self._mode == "recipe" else 1
            self._rebuild()
        elif self._phase == "values":
            self._phase = "param" if self._mode == "custom" else "recipe"
            self._sel = 0
            self._rebuild()
        elif self._phase in ("values_explicit", "values_range"):
            self._phase = "values"
            self._sel = 0
            self._rebuild()
        elif self._phase == "num_runs":
            if self._mode == "recipe":
                self._phase = "recipe"
                self._sel = self._recipe_idx
            else:
                self._phase = "values_explicit"
                self._sel = 0
            self._rebuild()
        elif self._phase == "netlist":
            self._phase = "num_runs"
            self._sel = 0
            self._rebuild()
        elif self._phase == "delete":
            self._phase = "netlist"
            self._sel = self._netlist_idx
            self._rebuild()
        elif self._phase == "summary":
            self._phase = "delete"
            self._sel = 0
            self._rebuild()

    def _handle_enter(self):
        nets = available_netlists()
        if self._phase == "type":
            self._mode = "recipe" if self._sel == 0 else "custom"
            self._phase = "recipe" if self._mode == "recipe" else "param"
            self._sel = 0
            self._rebuild()
        elif self._phase == "recipe":
            self._recipe_idx = self._sel
            r = PREDEFINED_RECIPES[self._recipe_idx]
            self._sweep_param = r["param"]
            self._sweep_values = r["values"]
            self._phase = "num_runs"
            self._sel = 0
            self._rebuild()
        elif self._phase == "param":
            self._sweep_param = self._param_list[self._sel][0]
            self._phase = "values"
            self._sel = 0
            self._rebuild()
        elif self._phase == "values":
            if self._sel == 0:
                self._phase = "values_explicit"
                self._edit_mode = True
                self._edit_buffer = ",".join(str(v) for v in self._sweep_values)
            else:
                self._phase = "values_range"
                self._edit_mode = True
                self._edit_buffer = "64,256,64"
            self._rebuild()
        elif self._phase == "values_explicit":
            self._edit_mode = True
            self._edit_buffer = ",".join(str(v) for v in self._sweep_values)
            self._rebuild()
        elif self._phase == "values_range":
            self._edit_mode = True
            self._edit_buffer = ",".join(str(v) for v in self._sweep_values)
            self._rebuild()
        elif self._phase == "num_runs":
            self._edit_mode = True
            self._edit_buffer = str(self._num_runs)
            self._rebuild()
        elif self._phase == "netlist":
            self._netlist_idx = self._sel
            self._phase = "delete"
            self._sel = 0
            self._rebuild()
        elif self._phase == "delete":
            self._delete_artifacts = (self._sel == 1)
            self._phase = "summary"
            self._sel = 0
            self._rebuild()

    def _handle_next(self):
        if self._phase in ("values_explicit", "values_range"):
            self._phase = "num_runs"
            self._sel = 0
            self._rebuild()
        elif self._phase == "num_runs":
            self._phase = "netlist"
            self._sel = 0
            self._rebuild()
        elif self._phase == "netlist":
            self._phase = "delete"
            self._sel = 0
            self._rebuild()
        elif self._phase == "summary":
            nets = available_netlists()
            base_config = {
                "netlist_path": f"netlists/{nets[self._netlist_idx]}.json" if nets else "netlists/xerox.json",
                "num_rows": 128, "num_cols": 128,
                "hidden_dim": 128, "num_hidden": 3, "hidden_channels_e": 128, "num_layers_e": 3,
                "gamma": 0.99, "clip_epsilon": 0.2, "value_loss_coef": 0.5, "entropy_coef": 0.8,
                "num_iterations": 1000, "lr": 0.0003,
                "vanilla_iterations": 100, "graph_ppo_iterations": 100,
                "mode": "full_pipeline",
                "reward_predictor": {"hidden_channels_r": 128, "num_layers_r": 3, "batch_size": 32, "num_epochs": 100},
                "graph_ppo": {"num_iterations": 100, "freeze_encoder": False},
            }
            from .param_editor import VimFormScreen
            sp = self._sweep_param
            sv = self._sweep_values
            nr = self._num_runs
            da = self._delete_artifacts

            def _launch(config):
                self.app.push_screen(
                    ExperimentProgressScreen(
                        base_config=config, sweep_param=sp,
                        sweep_values=sv, num_runs=nr, delete_artifacts=da,
                    )
                )

            self.app.push_screen(
                VimFormScreen(
                    mode="full_pipeline", config=base_config,
                    title="Base Configuration for Experiment",
                    submit_label="Start Experiment", submit_callback=_launch,
                )
            )

    @staticmethod
    def _parse_values(s: str) -> list:
        parts = [x.strip() for x in s.split(",")]
        if len(parts) == 3:
            try:
                from_v, to_v, step = float(parts[0]), float(parts[1]), float(parts[2])
                result = []
                v = from_v
                while v <= to_v + 1e-9:
                    result.append(int(v) if v == int(v) else round(v, 6))
                    v += step
                if result:
                    return result
            except ValueError:
                pass
        return [int(x) if x.lstrip("-").isdigit() else float(x) for x in parts]


class ExperimentProgressScreen(Screen):

    def __init__(self, base_config: dict, sweep_param: str, sweep_values: list, num_runs: int = 3, delete_artifacts: bool = False):
        super().__init__()
        self.base_config = base_config
        self.sweep_param = sweep_param
        self.sweep_values = sweep_values
        self.num_runs = num_runs
        self.delete_artifacts = delete_artifacts
        self._stop_event = threading.Event()
        self._thread = None
        self._results = []

    def compose(self) -> ComposeResult:
        yield Static("Experiment in Progress")
        yield Static("")
        with Vertical(id="progress-container"):
            yield Static("", id="exp-status")
            yield ProgressBar(total=100, id="exp-pbar")
            with Vertical(id="log-container"):
                yield RichLog(id="exp-log", highlight=True, max_lines=500)
            with Horizontal(id="action-bar"):
                yield Button("Cancel", id="cancel-btn")
        yield footer_bar("c Cancel  Esc Back")

    def on_mount(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        def cb(msg):
            if self._stop_event.is_set():
                return
            self.app.call_from_thread(self._on_progress, msg)

        try:
            results = run_experiment(
                base_config=self.base_config,
                sweep_param=self.sweep_param,
                sweep_values=self.sweep_values,
                num_runs=self.num_runs,
                delete_artifacts=self.delete_artifacts,
                stop_event=self._stop_event,
                progress_callback=cb,
            )
            self._results = results
            self.app.call_from_thread(self._on_done)
        except Exception as e:
            import traceback
            self.app.call_from_thread(self._log, f"Error: {e}\n{traceback.format_exc()}")

    def _on_progress(self, msg):
        kind = msg.get("kind", "")
        if kind == "run_start":
            self.query_one("#exp-status").update(f"Run {msg['idx']+1}/{msg['total']}: {msg['label']}")
            self._log(f"Starting: {msg['label']}")
        elif kind == "run_done":
            total = msg.get("total", 1)
            done = msg.get("idx", 0) + 1
            self.query_one("#exp-pbar").progress = min(100, done / total * 100)
            self._log(f"Done: HPWL={msg.get('hpwl', '?')} ({done}/{total})")
        elif kind == "log":
            self._log(msg.get("text", ""))

    def _log(self, text):
        try:
            self.query_one("#exp-log").write(text)
        except Exception:
            pass

    def _on_done(self):
        self._log("Experiment complete!")
        self.app.pop_screen()
        from .experiment_screens import ExperimentResultsScreen
        self.app.push_screen(ExperimentResultsScreen(self._results, self.sweep_param, self.sweep_values))

    def on_button_pressed(self, event):
        if event.button.id == "cancel-btn":
            self.action_cancel()

    def on_key(self, event: events.Key) -> None:
        if event.key == "c":
            self.action_cancel()
            event.stop()
        elif event.key == "escape":
            self.app.pop_screen()
            event.stop()

    def action_cancel(self):
        self._stop_event.set()
        self._log("Cancelling...")


class ExperimentResultsScreen(Screen):

    def __init__(self, results, sweep_param, sweep_values):
        super().__init__()
        self.results = results
        self.sweep_param = sweep_param
        self.sweep_values = sweep_values

    def compose(self) -> ComposeResult:
        yield Static("Experiment Complete")
        yield Static("")
        with Vertical(id="complete-container"):
            best_hpwl = min((r.get("hpwl", float("inf")) for r in self.results if r.get("hpwl") is not None), default=None)
            if best_hpwl is not None:
                best_result = next((r for r in self.results if r.get("hpwl") == best_hpwl), None)
                best_val = best_result["value"] if best_result else "?"
                yield Static(f"Best: {self.sweep_param}={best_val} -> HPWL={best_hpwl:.2f}")
            else:
                yield Static("Experiment completed (no HPWL data)")
            yield Static("")
            yield Static("Report saved to:")
            report_path = self.results[0].get("report_path") if self.results else None
            if report_path:
                yield Static(f"  {report_path}")
            yield Static("")
            yield Static("Press Esc for menu, q to quit")
        yield footer_bar("Esc Back  q Quit")

    def action_go_back(self):
        from .main_menu import MainMenuScreen
        self.app.switch_screen(MainMenuScreen())

    def on_key(self, event: events.Key) -> None:
        if event.key in ("escape", "m", "q"):
            self.action_go_back()
            event.stop()