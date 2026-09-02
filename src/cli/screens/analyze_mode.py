# Analyze mode - generate plots from artifact runs.

from datetime import datetime
from pathlib import Path

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static
from textual.containers import Vertical
from textual import events

from .._utils import artifact_runs, available_netlists, footer_bar
from ..plots import learning_curve, placement_compare, scale_generalize

ANALYSIS_DIR = Path("analysis")


class AnalyzeModeScreen(Screen):

    PLOT_TYPES = [
        ("learning-curve", "Learning Curve"),
        ("placement-compare", "Placement Compare"),
        ("scale-generalize", "Scale Generalize"),
    ]

    def __init__(self):
        super().__init__()
        self._phase = "plot_type"
        self._sel = 0
        self._plot_type = "learning-curve"
        self._run_idx = 0
        self._before_idx = 0
        self._after_idx = 0
        self._netlist_idx = 0
        self._gen_netlist_idxs = []
        self._widgets = []

    def compose(self) -> ComposeResult:
        yield Static("Generate Plots")
        yield Static("")
        with Vertical(id="wizard-container"):
            for _ in range(20):
                w = Static("")
                self._widgets.append(w)
                yield w
        yield footer_bar("j Down  k Up  Enter Select  g Generate  Esc Back")

    def on_mount(self) -> None:
        self._rebuild()

    def _rebuild(self):
        runs = artifact_runs() or ["(no runs)"]
        nets = available_netlists() or ["(no netlists)"]
        lines = []

        if self._phase == "plot_type":
            lines.append(("header", "Plot type:"))
            for i, (key, label) in enumerate(self.PLOT_TYPES):
                p = ">" if i == self._sel else " "
                t = "option" if i == self._sel else "text"
                lines.append((t, f"  {p} {label}"))

        elif self._phase == "before":
            lines.append(("header", "Before run (Vanilla PVN):"))
            for i, r in enumerate(runs):
                p = ">" if i == self._sel else " "
                t = "option" if i == self._sel else "text"
                lines.append((t, f"  {p} {r}"))

        elif self._phase == "after":
            lines.append(("header", "After run (Graph PPO):"))
            for i, r in enumerate(runs):
                p = ">" if i == self._sel else " "
                t = "option" if i == self._sel else "text"
                lines.append((t, f"  {p} {r}"))

        elif self._phase == "netlist":
            lines.append(("header", "Netlist:"))
            for i, n in enumerate(nets):
                p = ">" if i == self._sel else " "
                t = "option" if i == self._sel else "text"
                lines.append((t, f"  {p} {n}"))

        elif self._phase == "run":
            gppo_runs = [r for r in runs if (Path("artifacts") / r / "graph_ppo_final.pt").exists()]
            if not gppo_runs:
                gppo_runs = ["(no graph ppo runs found)"]
            lines.append(("header", "Run (must contain graph_ppo_final.pt):"))
            for i, r in enumerate(gppo_runs):
                p = ">" if i == self._sel else " "
                t = "option" if i == self._sel else "text"
                lines.append((t, f"  {p} {r}"))

        elif self._phase == "gen_netlists":
            lines.append(("header", "Select netlists to evaluate:"))
            for i, n in enumerate(nets):
                selected = " [x]" if i in self._gen_netlist_idxs else " [ ]"
                p = ">" if i == self._sel else " "
                t = "option" if i == self._sel else "text"
                lines.append((t, f"  {p}{selected} {n}"))

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

    def _max_sel(self, runs, nets) -> int:
        if self._phase == "plot_type":
            return len(self.PLOT_TYPES)
        elif self._phase == "run":
            gppo_runs = [r for r in runs if (Path("artifacts") / r / "graph_ppo_final.pt").exists()]
            return max(len(gppo_runs), 1)
        elif self._phase in ("before", "after"):
            return len(runs)
        elif self._phase in ("netlist", "gen_netlists"):
            return len(nets)
        return 1

    def on_key(self, event: events.Key) -> None:
        runs = artifact_runs()
        nets = available_netlists()
        max_sel = self._max_sel(runs, nets)

        if event.key == "j":
            self._sel = (self._sel + 1) % max_sel
            self._rebuild()
            event.stop()
        elif event.key == "k":
            self._sel = (self._sel - 1) % max_sel
            self._rebuild()
            event.stop()
        elif event.key == "escape":
            if self._phase == "plot_type":
                self.app.pop_screen()
            elif self._phase == "before":
                self._phase = "plot_type"; self._sel = 1; self._rebuild()
            elif self._phase == "after":
                self._phase = "before"; self._sel = self._before_idx; self._rebuild()
            elif self._phase == "netlist":
                self._phase = "after"; self._sel = self._after_idx; self._rebuild()
            elif self._phase == "run":
                self._phase = "plot_type"; self._sel = 0; self._rebuild()
            elif self._phase == "gen_netlists":
                self._phase = "run"; self._sel = self._run_idx; self._rebuild()
            event.stop()
        elif event.key == "enter":
            self._handle_enter(runs, nets)
            event.stop()
        elif event.key == "g" and self._phase == "gen_netlists":
            self._generate_scale(runs, nets)
            event.stop()

    def _handle_enter(self, runs, nets):
        if self._phase == "plot_type":
            entry = self.PLOT_TYPES[self._sel]
            self._plot_type = entry[0]
            pt = entry[0]
            if pt == "learning-curve":
                self._phase = "run"; self._sel = 0
            elif pt == "placement-compare":
                self._phase = "before"; self._sel = 0
            elif pt == "scale-generalize":
                self._phase = "run"; self._sel = 0
                self._gen_netlist_idxs = list(range(len(nets))) if nets else []
            self._rebuild()

        elif self._phase == "before":
            self._before_idx = self._sel
            self._phase = "after"; self._sel = self._after_idx
            self._rebuild()

        elif self._phase == "after":
            self._after_idx = self._sel
            self._phase = "netlist"; self._sel = self._netlist_idx
            self._rebuild()

        elif self._phase == "netlist":
            self._netlist_idx = self._sel
            self._generate_placement(runs, nets)

        elif self._phase == "run":
            self._run_idx = self._sel
            if self._plot_type == "scale-generalize":
                gppo_runs = [r for r in runs if (Path("artifacts") / r / "graph_ppo_final.pt").exists()]
                if self._sel < len(gppo_runs):
                    self._run_idx = runs.index(gppo_runs[self._sel])
                self._phase = "gen_netlists"; self._sel = 0
                self._rebuild()
            else:
                self._generate_learning(runs)

        elif self._phase == "gen_netlists":
            if self._sel < len(nets):
                if self._sel in self._gen_netlist_idxs:
                    self._gen_netlist_idxs.remove(self._sel)
                else:
                    self._gen_netlist_idxs.append(self._sel)
                self._rebuild()

    def _ts(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _generate_learning(self, runs):
        if not runs or runs[0] == "(no runs)":
            self._notify("No runs available", severity="error")
            return
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            fig = learning_curve(Path("artifacts") / runs[self._run_idx])
            path = ANALYSIS_DIR / f"{runs[self._run_idx]}_learning_curve_{self._ts()}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            self._notify(f"Saved: {path}")
        except Exception as e:
            self._notify(f"Error: {e}", severity="error")

    def _generate_placement(self, runs, nets):
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            fig = placement_compare(
                Path("artifacts") / runs[self._before_idx],
                Path("artifacts") / runs[self._after_idx],
                nets[self._netlist_idx], 128, 128,
            )
            path = ANALYSIS_DIR / f"{runs[self._before_idx]}_vs_{runs[self._after_idx]}_placement_{self._ts()}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            self._notify(f"Saved: {path}")
        except Exception as e:
            self._notify(f"Error: {e}", severity="error")

    def _generate_scale(self, runs, nets):
        if not self._gen_netlist_idxs:
            self._notify("No netlists selected", severity="error")
            return
        if self._run_idx >= len(runs):
            self._notify("Invalid run selection", severity="error")
            return
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            names = [nets[i] for i in self._gen_netlist_idxs]
            paths = [f"netlists/{n}.json" for n in names]
            fig = scale_generalize(
                Path("artifacts") / runs[self._run_idx],
                paths, names, 128, 128,
            )
            path = ANALYSIS_DIR / f"{runs[self._run_idx]}_scale_generalize_{self._ts()}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            self._notify(f"Saved: {path}")
        except Exception as e:
            self._notify(f"Error: {e}", severity="error")

    def _notify(self, msg, severity="information"):
        try:
            self.app.notify(msg, severity=severity)
        except Exception:
            print(msg)