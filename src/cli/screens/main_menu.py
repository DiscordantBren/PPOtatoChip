from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static

from .._utils import footer_bar


class MainMenuScreen(Screen):
    """Main menu with Train, Analyze, Experiment, Exit."""

    BINDINGS = [
        ("q", "quit_app", "Quit"),
        ("1", "train", "Train"),
        ("2", "analyze", "Analyze"),
        ("3", "experiment", "Experiment"),
    ]

    def compose(self) -> ComposeResult:
        logo = """
██████╗ ██████╗  ██████╗ ████████╗ █████╗ ████████╗ ██████╗
██╔══██╗██╔══██╗██╔═══██╗╚══██╔══╝██╔══██╗╚══██╔══╝██╔═══██╗
██████╔╝██████╔╝██║   ██║   ██║   ███████║   ██║   ██║   ██║
██╔═══╝ ██╔═══╝ ██║   ██║   ██║   ██╔══██║   ██║   ██║   ██║
██║     ██║     ╚██████╔╝   ██║   ██║  ██║   ██║   ╚██████╔╝
╚═╝     ╚═╝      ╚═════╝    ╚═╝   ╚═╝  ╚═╝   ╚═╝    ╚═════╝

 ██████╗██╗  ██╗██╗██████╗
██╔════╝██║  ██║██║██╔══██╗
██║     ███████║██║██████╔╝
██║     ██╔══██║██║██╔═══╝
╚██████╗██║  ██║██║██║
 ╚═════╝╚═╝  ╚═╝╚═╝╚═╝"""
        yield Static(logo)
        yield Static("  Reinforcement Learning for Chip Placement")
        yield Static("")
        yield Static("  1. Train Pipeline")
        yield Static("  2. Analyze Runs")
        yield Static("  3. Run Experiment")
        yield Static("")
        yield Static("  Press a number or q to quit")
        yield footer_bar("1 Train  2 Analyze  3 Experiment  q Quit")

    def action_train(self) -> None:
        from .train_mode import TrainModeScreen
        self.app.push_screen(TrainModeScreen())

    def action_analyze(self) -> None:
        from .analyze_mode import AnalyzeModeScreen
        self.app.push_screen(AnalyzeModeScreen())

    def action_experiment(self) -> None:
        from .experiment_screens import ExperimentSetupScreen
        self.app.push_screen(ExperimentSetupScreen())

    def action_quit_app(self) -> None:
        self.app.exit()