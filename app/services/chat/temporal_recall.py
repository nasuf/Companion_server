"""Compatibility import; implementation and mutable state live in app.services.time_domain.recall."""
from importlib import import_module
import sys

sys.modules[__name__] = import_module("app.services.time_domain.recall")
