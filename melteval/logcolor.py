"""This is just for highlighting lines that stand out from the rest of an eval's output.
"""

from __future__ import annotations

import logging


BOLD_YELLOW = "\x1b[1;33m"
RESET = "\x1b[0m"


def log_skipped_sample(logger: logging.Logger, reason: str) -> None:
    """Log, in colour, that one sample was failed on purpose rather than evaluated."""
    logger.error(f"{BOLD_YELLOW}[SAMPLE SKIPPED] {reason}{RESET}")
