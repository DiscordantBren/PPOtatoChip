# Shared helpers - netlists, artifact runs, footer bar.

from pathlib import Path

from textual.widgets import Static


def available_netlists() -> list[str]:
    netlist_dir = Path("netlists")
    if not netlist_dir.exists():
        return []
    return sorted([p.stem for p in netlist_dir.glob("*.json")])


def artifact_runs() -> list[str]:
    artifacts_dir = Path("artifacts")
    if not artifacts_dir.exists():
        return []
    return sorted(
        [p.name for p in artifacts_dir.iterdir() if p.is_dir()],
        reverse=True,
    )


def footer_bar(text: str) -> Static:
    bar = Static(text, classes="footer-bar")
    return bar