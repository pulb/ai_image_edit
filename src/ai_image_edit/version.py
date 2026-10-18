# SPDX-License-Identifier: GPL-3.0-or-later
"""
The app's version, as setuptools-scm derived it from the git tags when the
package was built (see pyproject.toml): "0.2.0" on a release, "0.2.1.dev4+g1a2b3c4"
after it. "unknown" if the package is not installed.
"""
from importlib import metadata


def installed_version() -> str:
    try:
        return metadata.version("ai-image-edit")
    except metadata.PackageNotFoundError:
        return "unknown"


VERSION = installed_version()
