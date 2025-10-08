"""
utils/portfolio_utils.py

This wrapper module re-exports all public symbols from the top-level
``portfolio_utils`` module.  It allows client code to import the
utilities from the ``utils`` namespace (e.g. ``from utils.portfolio_utils
import process_file``) without breaking existing imports that refer to
``portfolio_utils`` directly.  The wrapper inspects the ``__all__``
attribute of the underlying module to determine which names to re-export.
"""

from portfolio_utils import *  # noqa: F401,F403  re-export everything defined in __all__
