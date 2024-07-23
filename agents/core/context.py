import os
from typing import Any, Dict, List

from ..config import GlobalConfig


class GlobalContext:
    def __init__(self, maximum_chars=5000, db_path: str = None):
        if db_path is None:
            db_path = GlobalConfig.get_resolved_db_path()
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.history: List[Dict[str, Any]] = []
        self.maximum_chars = maximum_chars

    def add(
        self,
        agent_name: str,
        task: str,
        result: Any,
        status: str,
        start_time: str,
        end_time: str,
    ):
        self.history.append(
            {
                "agent": agent_name,
                "task": task,
                "result": result,
                "status": status,
                "start_time": start_time,
                "end_time": end_time,
            }
        )

    def get_context_string(self):
        return "\n".join(
            [
                f"{entry['agent']}: {entry['task']} -> {entry['result']}"
                for entry in self.history
            ]
        )[-1 * self.maximum_chars :]

    def __del__(self):
        self.end_run()

    def end_run(self):
        pass
