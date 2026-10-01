# SPDX-License-Identifier: GPL-3.0-or-later
"""Cross-cutting exception type shared by every model backend and the UI."""


class GenerationError(Exception):
    """Raised by a model backend for any user-facing failure during generation."""
