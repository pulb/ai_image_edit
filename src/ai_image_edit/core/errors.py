# SPDX-License-Identifier: GPL-3.0-or-later
"""Cross-cutting exception types shared by every model and the UI."""


class GenerationError(Exception):
    """Raised by a model for any user-facing failure during generation."""


class GenerationCancelled(GenerationError):
    """Raised by generate() when the user cancelled the generation."""


class StartupError(Exception):
    """
    A failure that keeps the app from starting. The message goes to the log;
    the error page shows only `area` and `detail`, which are public, so they
    must be fixed texts (setting names at most, never values).
    """

    def __init__(self, area: str, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.area = area
        self.detail = detail
