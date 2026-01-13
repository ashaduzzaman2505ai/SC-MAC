# tests/test_worker.py
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from src.agents.worker import LogicWorker

def test_run():
    # Use Llama-3-8B (or a 2026 equivalent like Llama-4-mini)
    worker = LogicWorker("meta-llama/Meta-Llama-3-8B-Instruct")
    result = worker.reasoning_loop("If all A are B and some B are C, are all A necessarily C?")
    print("\nFinal Output:\n", result)

if __name__ == "__main__":
    test_run()
