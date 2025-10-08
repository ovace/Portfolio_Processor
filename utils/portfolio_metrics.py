"""
utils/portfolio_metrics.py

This wrapper module re-exports all public symbols from the top-level
``portfolio_metrics`` module.  It allows client code to import the
utilities from the ``utils`` namespace (e.g. ``from utils.portfolio_metrics
import process_file``) without breaking existing imports that refer to
``portfolio_metrics`` directly.  The wrapper inspects the ``__all__``
attribute of the underlying module to determine which names to re-export.
"""

from portfolio_metrics import *  # noqa: F401,F403  re-export everything defined in __all__
