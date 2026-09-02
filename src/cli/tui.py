from textual.app import App

from .screens.main_menu import MainMenuScreen


class PPOtatoChipApp(App):
    CSS_PATH = "tui.css"
    SCREENS = {"main_menu": MainMenuScreen}
    TITLE = "PPOtatoChip"
    SUB_TITLE = "RL-based Chip Placement Optimization"
    BINDINGS = []

    def on_mount(self) -> None:
        self.push_screen("main_menu")


def main():
    app = PPOtatoChipApp()
    app.run()


if __name__ == "__main__":
    main()