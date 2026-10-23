"""Deprecated shim: re-exports the new app.services modules.

Kept for one release so external code importing `caller` keeps working.
New code should import from app.services.predictor / video_stream / state.
"""

import warnings

warnings.warn("caller.py is deprecated; use app.services.* instead",
              DeprecationWarning, stacklevel=2)

from app.services.predictor import Predictor  # noqa: F401
from app.services.state import get_state  # noqa: F401
from app.services.video_stream import video_stream as _video_stream  # noqa: F401


def test(info):
    print(info)


def videoStream(camera_index=0):  # noqa: N802 (legacy name)
    yield from _video_stream(Predictor.peek(), camera_index)
