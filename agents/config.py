import os


class GlobalConfig:
    LIVE_VERBOSITY_LEVEL = 1
    DEFAULT_DB_PATH = "~/.cache/multi_agent_system/db.sqlite3"

    @classmethod
    def get_resolved_db_path(cls) -> str:
        """
        Resolve the default database path to an absolute path.
        """
        return os.path.expanduser(cls.DEFAULT_DB_PATH)
