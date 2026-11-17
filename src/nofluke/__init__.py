"""nofluke: ask/tell search over slow, noisy evaluations that run outside Python, without rewarding flukes.

Public API: `Study` (create/open a study folder, ask, tell, best), `Space`, `optimize` (local evaluator loop),
`NoflukeError` and `__version__`. The command line is `nofluke.cli.main`.
"""

from __future__ import annotations

from importlib import metadata

try:
    __version__ = metadata.version("nofluke")
except metadata.PackageNotFoundError:
    __version__ = "0+unknown"

from nofluke.errors import NoflukeError  # noqa: E402
from nofluke.runner import optimize  # noqa: E402
from nofluke.space import Space  # noqa: E402
from nofluke.study import Study  # noqa: E402

__all__ = ["NoflukeError", "Space", "Study", "__version__", "optimize"]
