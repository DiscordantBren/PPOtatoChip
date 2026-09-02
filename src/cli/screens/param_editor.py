"""
Vim-style parameter form.
j/k moves between fields. Enter to edit. Characters type into the value.
Esc to cancel. s to submit.
"""

from pathlib import Path

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static
from textual.containers import Vertical
from textual import events

from .._utils import available_netlists, footer_bar


class VimFormScreen(Screen):

    def __init__(self, mode: str = "vanilla_pvn", config: dict | None = None,
                 title: str = "", submit_label: str = "Start",
                 submit_callback=None):
        super().__init__()
        self.mode = mode
        self.base_config = config or {}
        self._custom_title = title
        self.submit_label = submit_label
        self._submit_callback = submit_callback
        self._fields = []
        self._sel = 0
        self._editing = False
        self._edit_original = ""

    def _add_field(self, label: str, value: str, field_id: str, ftype: str = "text", options: list = None):
        self._fields.append({
            "label": label,
            "value": str(value),
            "id": field_id,
            "type": ftype,
            "options": options or [],
            "widget": None,
        })

    def compose(self) -> ComposeResult:
        title = self._custom_title or f"Training: {self.mode.replace('_', ' ').title()}"
        yield Static(title)
        yield Static("")
        with Vertical(id="form-container"):
            pass
        yield footer_bar("j Down  k Up  Enter Edit  Esc Back  s Start  Tab Next")

    def on_mount(self) -> None:
        c = self.base_config

        from .._utils import available_netlists
        nets = available_netlists()
        current_np = c.get("netlist_path", "netlists/xerox.json")
        current_stem = Path(current_np).stem if current_np else "xerox"
        net_opts = [(n, f"netlists/{n}.json") for n in nets]
        self._add_field("netlist", current_np, "netlist_path", "select", options=net_opts)
        self._add_field("rows", str(c.get("num_rows", 128)), "num_rows", "int")
        self._add_field("cols", str(c.get("num_cols", 128)), "num_cols", "int")
        self._add_field("mlp_dim", str(c.get("hidden_dim", 128)), "hidden_dim", "int")
        self._add_field("mlp_layers", str(c.get("num_hidden", 3)), "num_hidden", "int")
        self._add_field("gnn_ch", str(c.get("hidden_channels_e", 128)), "hidden_channels_e", "int")
        self._add_field("gnn_layers", str(c.get("num_layers_e", 3)), "num_layers_e", "int")
        self._add_field("gamma", str(c.get("gamma", 0.99)), "gamma", "float")
        self._add_field("clip_eps", str(c.get("clip_epsilon", 0.2)), "clip_epsilon", "float")
        self._add_field("v_coef", str(c.get("value_loss_coef", 0.5)), "value_loss_coef", "float")
        self._add_field("entropy", str(c.get("entropy_coef", 0.8)), "entropy_coef", "float")
        self._add_field("iterations", str(c.get("num_iterations", 1000)), "num_iterations", "int")
        self._add_field("lr", str(c.get("lr", 0.0003)), "lr", "float")

        if self.mode in ("reward_predictor", "full_pipeline"):
            self._add_field("rp_mlp_ch", str(c.get("hidden_channels_r", 128)), "hidden_channels_r", "int")
            self._add_field("rp_mlp_layers", str(c.get("num_layers_r", 3)), "num_layers_r", "int")
            self._add_field("rp_batch", str(c.get("batch_size", 32)), "batch_size", "int")
            self._add_field("rp_epochs", str(c.get("num_epochs", 100)), "num_epochs", "int")

        if self.mode == "reward_predictor":
            from .._utils import artifact_runs
            placement_options = []
            for run_name in artifact_runs():
                pp = Path("artifacts") / run_name / "placements.jsonl"
                if pp.exists():
                    placement_options.append((run_name, str(pp)))
            if not placement_options:
                self._add_field("placements_path", c.get("placements_path", "(not found)"), "placements_path", "path")
            else:
                self._add_field("placements_path", c.get("placements_path", placement_options[0][1]),
                                "placements_path", "select", options=placement_options)

        if self.mode in ("graph_ppo",):
            from .._utils import artifact_runs
            encoder_options = []
            for run_name in artifact_runs():
                ep = Path("artifacts") / run_name / "encoder.pt"
                if ep.exists():
                    encoder_options.append((run_name, str(ep)))
            current_ep = str(c.get("pretrained_encoder_path", "")) if c.get("pretrained_encoder_path") else ""
            if not encoder_options:
                self._add_field("encoder_path", current_ep or "(none found, type path)", "pretrained_encoder_path", "path")
            else:
                self._add_field("encoder_path", current_ep or encoder_options[0][1], "pretrained_encoder_path", "select",
                               options=encoder_options)
            self._add_field("freeze", str(c.get("freeze_encoder", False)), "freeze_encoder", "select",
                           options=[("False", False), ("True", True)])
        elif self.mode == "full_pipeline":
            self._add_field("freeze", str(c.get("freeze_encoder", False)), "freeze_encoder", "select",
                           options=[("False", False), ("True", True)])

        if self.mode == "full_pipeline":
            self._add_field("vanilla_iter", str(c.get("vanilla_iterations", 100)), "vanilla_iterations", "int")
            self._add_field("graph_iter", str(c.get("graph_ppo_iterations", 100)), "graph_ppo_iterations", "int")

        container = self.query_one("#form-container")
        for f in self._fields:
            w = Static("")
            f["widget"] = w
            container.mount(w)

        self._update_all()

    def _render_value(self, f: dict) -> str:
        return f"  {f['label']}: {f['value']}"

    def _update_all(self):
        for i, f in enumerate(self._fields):
            w = f["widget"]
            display = self._render_value(f)
            f["_display"] = display
            if i == self._sel and not self._editing:
                w.styles.background = "#ffffff"
                w.styles.color = "#000000"
            else:
                w.styles.background = "#000000"
                w.styles.color = "#ffffff"
            w.update(display)

    def _start_edit(self):
        f = self._fields[self._sel]
        self._editing = True
        self._edit_original = f["value"]
        if f["type"] == "select":
            opts = f["options"]
            current = f["value"]
            lines = []
            for opt_label, opt_val in opts:
                p = ">" if str(opt_val) == current else " "
                lines.append(f"    {p} {opt_label}")
            display = f"  {f['label']}: [{current}]\n" + "\n".join(lines)
        else:
            display = f"  {f['label']}: [{f['value']}]"
        f["_display"] = display
        f["widget"].update(display)
        f["widget"].styles.background = "#000000"
        f["widget"].styles.color = "#ffffff"

    def _confirm_edit(self):
        f = self._fields[self._sel]
        self._editing = False
        self._update_all()

    def _cancel_edit(self):
        f = self._fields[self._sel]
        f["value"] = self._edit_original
        self._editing = False
        self._update_all()

    def _gather(self) -> dict:
        c = {"mode": self.mode}
        for f in self._fields:
            key = f["id"]
            val = f["value"]
            typ = f["type"]
            if typ == "int":
                c[key] = int(val) if val else 0
            elif typ == "float":
                c[key] = float(val) if val else 0.0
            elif typ == "select":
                for opt_label, opt_val in f["options"]:
                    if str(opt_val) == val:
                        c[key] = opt_val
                        break
                else:
                    c[key] = val
            else:
                c[key] = val

        if self.mode == "full_pipeline":
            c["reward_predictor"] = {
                "hidden_channels_r": c.get("hidden_channels_r", 128),
                "num_layers_r": c.get("num_layers_r", 3),
                "batch_size": c.get("batch_size", 32),
                "num_epochs": c.get("num_epochs", 100),
            }
            c["graph_ppo"] = {
                "num_iterations": c.pop("graph_ppo_iterations", 100),
                "freeze_encoder": c.get("freeze_encoder", False),
            }
        return c

    def action_next_field(self):
        if not self._fields:
            return
        if self._editing:
            f = self._fields[self._sel]
            if f["type"] == "select":
                self._cycle_select(1)
            return
        self._sel = (self._sel + 1) % len(self._fields)
        self._update_all()

    def action_prev_field(self):
        if not self._fields:
            return
        if self._editing:
            f = self._fields[self._sel]
            if f["type"] == "select":
                self._cycle_select(-1)
            return
        self._sel = (self._sel - 1) % len(self._fields)
        self._update_all()

    def _cycle_select(self, direction: int):
        f = self._fields[self._sel]
        opts = f["options"]
        current = f["value"]
        idx = 0
        for i, (_, ov) in enumerate(opts):
            if str(ov) == current:
                idx = i
                break
        idx = (idx + direction) % len(opts)
        f["value"] = str(opts[idx][1])
        current = f["value"]
        lines = []
        for opt_label, opt_val in opts:
            p = ">" if str(opt_val) == current else " "
            lines.append(f"    {p} {opt_label}")
        display = f"  {f['label']}: [{current}]\n" + "\n".join(lines)
        f["_display"] = display
        f["widget"].update(display)

    def on_key(self, event: events.Key) -> None:
        if not self._fields:
            return

        if self._editing:
            f = self._fields[self._sel]
            if f["type"] == "select":
                if event.key == "enter":
                    self._confirm_edit()
                    event.stop()
                elif event.key == "escape":
                    self._cancel_edit()
                    event.stop()
                elif event.key == "j":
                    self._cycle_select(1)
                    event.stop()
                elif event.key == "k":
                    self._cycle_select(-1)
                    event.stop()
                return

            if event.key == "enter":
                self._confirm_edit()
                event.stop()
            elif event.key == "escape":
                self._cancel_edit()
                event.stop()
            elif event.key == "j":
                self._confirm_edit()
                self.action_next_field()
                event.stop()
            elif event.key == "k":
                self._confirm_edit()
                self.action_prev_field()
                event.stop()
            elif event.key == "backspace":
                f["value"] = f["value"][:-1]
                display = f"  {f['label']}: [{f['value']}]"
                f["_display"] = display
                f["widget"].update(display)
                event.stop()
            elif event.key == "space":
                f["value"] += " "
                display = f"  {f['label']}: [{f['value']}]"
                f["_display"] = display
                f["widget"].update(display)
                event.stop()
            elif len(event.key) == 1 and event.key.isprintable():
                f["value"] += event.key
                display = f"  {f['label']}: [{f['value']}]"
                f["_display"] = display
                f["widget"].update(display)
                event.stop()
        else:
            if event.key == "enter":
                self._start_edit()
                event.stop()
            elif event.key == "escape":
                self.app.pop_screen()
                event.stop()
            elif event.key == "s":
                config = self._gather()
                if self._submit_callback:
                    self._submit_callback(config)
                else:
                    from .train_progress import TrainProgressScreen
                    self.app.push_screen(TrainProgressScreen(config=config))
                event.stop()
            elif event.key == "j":
                self.action_next_field()
                event.stop()
            elif event.key == "k":
                self.action_prev_field()
                event.stop()