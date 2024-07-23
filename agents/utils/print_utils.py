from typing import Any, Dict

from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from ..config import GlobalConfig

# Updated color palette with good contrast for both dark and light modes
COLORS = {
    "background": "#2E3440",  # Dark background
    "text": "#D8DEE9",  # Light text for dark background
    "agent": "#88C0D0",  # Pastel blue
    "task": "#A3BE8C",  # Pastel green
    "result": "#EBCB8B",  # Pastel yellow
    "llm": "#B48EAD",  # Pastel purple
    "attempts": "#D08770",  # Pastel orange
}


def print_agent_details(
    agent_name: str, task: str, result: Any, llm_name: str, attempts: int
):
    if GlobalConfig.LIVE_VERBOSITY_LEVEL < 0:
        return

    console = Console()

    if GlobalConfig.LIVE_VERBOSITY_LEVEL == 0:
        return

    elif GlobalConfig.LIVE_VERBOSITY_LEVEL == 1:
        table = Table(show_header=False, box=None)
        table.add_row(
            f"[bold {COLORS['agent']}]🤖 Agent:[/bold {COLORS['agent']}] {agent_name}"
        )
        table.add_row(
            f"[{COLORS['task']}]⏩ Input:[/{COLORS['task']}] {task[:50]}{'...' if len(str(task)) > 50 else ''}"
        )
        table.add_row(
            f"[{COLORS['result']}]🗒️ Output:[/{COLORS['result']}] {str(result)[:50]}{'...' if len(str(result)) > 50 else ''}"
        )
        panel = Panel(
            table,
            expand=False,
            border_style=COLORS["text"],
            style=f"on {COLORS['background']}",
        )
        console.print(panel)

    elif GlobalConfig.LIVE_VERBOSITY_LEVEL >= 2:
        tree = Tree(
            f"[bold {COLORS['agent']}]🤖 Agent: {agent_name}[/bold {COLORS['agent']}]"
        )

        input_tree = tree.add(f"[{COLORS['task']}]⏩ Input:[/{COLORS['task']}]")
        for line in str(task).split("\n"):
            input_tree.add(Text(line, style=COLORS["text"]))

        result_tree = tree.add(f"[{COLORS['result']}]🗒️ Output:[/{COLORS['result']}]")
        if isinstance(result, dict):
            for key, value in result.items():
                result_tree.add(Text(f"{key}: {value}", style=COLORS["text"]))
        else:
            for line in str(result).split("\n"):
                result_tree.add(Text(line, style=COLORS["text"]))

        if GlobalConfig.LIVE_VERBOSITY_LEVEL >= 3:
            tree.add(Text(f"🧠 LLM: {llm_name}", style=COLORS["llm"]))
            tree.add(Text(f"🔄 Attempts: {attempts}", style=COLORS["attempts"]))

        panel = Panel(
            tree,
            expand=False,
            border_style=COLORS["text"],
            style=f"on {COLORS['background']}",
        )
        console.print(panel)
