"""Error type raised for every mistake a user of nofluke can make.

`NoflukeError` messages name the offending field, parameter or id. The command line prints them and exits with
status 2; anything else that escapes is a bug.
"""

from __future__ import annotations


class NoflukeError(Exception):
    """A user-facing error: bad configuration, bad tell, unusable study folder."""
