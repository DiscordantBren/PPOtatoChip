# Training results / completion screen.

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static
from textual.containers import Vertical
from textual import events

from .._utils import footer_bar


class TrainCompleteScreen(Screen):

    def __init__(self, result, config: dict):
        super().__init__()
        self.result = result
        self.config = config

    def compose(self) -> ComposeResult:
        yield Static("Training Complete")
        yield Static("")

        mode = self.config.get("mode", "?")
        yield Static(f"  Mode: {mode.replace('_', ' ').title()}")

        artifact_path = "?"
        if self.result:
            mode_r = self.result[0]
            if mode_r in ("vanilla_pvn",):
                if self.result[1]:
                    artifact_path = str(self.result[1].path)
            elif mode_r in ("full_pipeline", "graph_ppo", "reward_predictor"):
                if self.result[2]:
                    artifact_path = str(self.result[2].path)

        yield Static(f"  Artifacts: {artifact_path}")
        yield Static("")
        yield Static("  Press Esc or m for menu, q to quit")

        yield footer_bar("Esc Menu  m Menu  q Quit")

    def action_go_back(self) -> None:
        from .main_menu import MainMenuScreen
        self.app.switch_screen(MainMenuScreen())

    def on_key(self, event: events.Key) -> None:
        if event.key in ("escape", "m"):
            self.action_go_back()
            event.stop()
        elif event.key == "q":
            self.app.exit()
            event.stop()