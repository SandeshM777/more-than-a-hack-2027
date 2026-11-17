"""Our possession pipeline for More than a Hack 2027.

Everything is tuned and trained on external games (IDSSE). The provided match is only ever read
as input by a finished, frozen model.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import games  # noqa: E402  (repo-root module, needs the path above)

# External, openly licensed games live next to the match bundle, never in the repository.
EXTERNAL_DIR = os.environ.get("MTAH_EXTERNAL_DIR") or os.path.join(games.DATA_DIR, "external")
