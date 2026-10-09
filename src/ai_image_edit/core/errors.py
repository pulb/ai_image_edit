# SPDX-License-Identifier: GPL-3.0-or-later
"""Cross-cutting exception type shared by every model and the UI."""


class GenerationError(Exception):
    """Raised by a model for any user-facing failure during generation."""
