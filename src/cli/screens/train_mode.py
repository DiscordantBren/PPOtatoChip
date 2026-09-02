# Train mode selection

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static
from textual.containers import Vertical
from textual import events

from .._utils import footer_bar


TRAIN_MODES = [
    ("vanilla_pvn", "Vanilla PVN"),
    ("reward_predictor", "Reward Predictor"),
    ("graph_ppo", "Graph PPO"),
    ("full_pipeline", "Full Pipeline"),
]


class TrainModeScreen(Screen):

    def __init__(self):
        super().__init__()
        self._sel = 0
        self._widgets = []

    def compose(self) -> ComposeResult:
        yield Static("Select Training Mode")
        yield Static("")
        with Vertical(id="wizard-container"):
            for _ in range(10):
                w = Static("")
                self._widgets.append(w)
                yield w
        yield footer_bar("j Down  k Up  Enter Select  Esc Back")

    def on_mount(self) -> None:
        self._rebuild()

    def _rebuild(self):
        for i, w in enumerate(self._widgets):
            if i < len(TRAIN_MODES):
                key, label = TRAIN_MODES[i]
                p = ">" if i == self._sel else " "
                w.update(f"  {p} {label}")
                if i == self._sel:
                    w.styles.background = "#ffffff"
                    w.styles.color = "#000000"
                else:
                    w.styles.background = "#000000"
                    w.styles.color = "#ffffff"
            else:
                w.update("")

    def on_key(self, event: events.Key) -> None:
        if event.key == "j":
            self._sel = (self._sel + 1) % len(TRAIN_MODES)
            self._rebuild()
            event.stop()
        elif event.key == "k":
            self._sel = (self._sel - 1) % len(TRAIN_MODES)
            self._rebuild()
            event.stop()
        elif event.key == "enter":
            mode = TRAIN_MODES[self._sel][0]
            from .param_editor import VimFormScreen
            self.app.push_screen(VimFormScreen(mode=mode))
            event.stop()
        elif event.key == "escape":
            self.app.pop_screen()
            event.stop()