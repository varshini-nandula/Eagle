import os
import yaml
from pathlib import Path

class PolicyLoader:
    def __init__(self):
        self.policy_path = os.getenv("POLICY_PATH", "policies/default.yaml")
        self._cache = None
        self._last_mtime = 0

    def load_policy(self):
        path = Path(self.policy_path)

        if not path.exists():
            raise FileNotFoundError(f"Policy file not found: {path}")

        # 🔥 CHECK FILE CHANGE TIME
        mtime = path.stat().st_mtime

        # 🔥 RELOAD IF FILE CHANGED
        if self._cache is None or mtime != self._last_mtime:
            with open(path, "r") as f:
                self._cache = yaml.safe_load(f)
            self._last_mtime = mtime

        return self._cache