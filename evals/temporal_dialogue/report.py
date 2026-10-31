"""Historical entry point; implementation is evals.temporal.dialogue.report."""
from importlib import import_module
import sys

if __name__ == "__main__":
    from pathlib import Path
    import runpy
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    runpy.run_module("evals.temporal.dialogue.report", run_name="__main__")
else:
    _implementation = import_module("evals.temporal.dialogue.report")
    globals().update({key: value for key, value in vars(_implementation).items() if not key.startswith("__")})
    sys.modules[__name__] = _implementation
