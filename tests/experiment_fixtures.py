"""Importable spawn fixtures. Never import a provider or read credentials."""
import os
from pathlib import Path
import subprocess
import sys
import time
from agent_eval.types import AgentOutcome


class Agent:
    def __init__(self, *, log_path=None, fail=False, child_pid_path=None, corrupt_result=False):
        self.log_path, self.fail, self.child_pid_path = log_path, fail, child_pid_path
        self.corrupt_result = corrupt_result

    def run(self, prompt):
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as stream:
                stream.write(prompt + "\n")
        if self.child_pid_path:
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                     creationflags=0x08000000 if os.name == "nt" else 0)
            Path(self.child_pid_path).write_text(str(child.pid))
            time.sleep(30)
        if self.fail:
            raise RuntimeError("controlled failure")
        if self.corrupt_result:
            original = Path.write_bytes
            Path.write_bytes = lambda path, data: original(path, b"{")
        return AgentOutcome("42", True, "finished", 1, 7)


def build_agent(**kwargs):
    return Agent(**kwargs)


def adapt(value):
    return value


class SlowScorer:
    metric_names = ("slow",)
    def score(self, task, outcome):
        time.sleep(30)
        return {"slow": True}


class CrashScorer:
    metric_names = ("crash",)
    def score(self, task, outcome):
        raise ValueError("controlled score error")
