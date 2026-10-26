# SPDX-License-Identifier: GPL-3.0-or-later
"""
The page served instead of the app when it cannot start. It is public (there is
no login without a running app), so it shows only fixed texts: which part
failed and, for some failures, one line on what is missing. The details are in
the log.
"""
from nicegui import ui

from ai_image_edit.ui.building_blocks import apply_dark_theme
from ai_image_edit.ui.helpers import HOST, PORT
from ai_image_edit.ui.login import LOGO_URI

DOCS_URL = "https://github.com/pulb/ai_image_edit#deployments"

_CSS = """
.aie-error-text p { margin: 0; }
.aie-error-text a { color: inherit; text-decoration: underline; }
.aie-error-text code { color: inherit; background: rgba(0, 0, 0, 0.25); padding: 1px 6px; border-radius: 4px; }
"""


def run(area: str, detail: str = "") -> None:
    """Serves the error page until the app is stopped. `detail` is Markdown."""

    @ui.page("/")
    def error_page() -> None:
        ui.context.client.status_code = 503  # the page is shown, but the service is not available
        apply_dark_theme()
        ui.add_css(_CSS)
        with ui.card().classes("absolute-center items-stretch bg-negative text-white").style("width: 512px; max-width: 90vw"):
            ui.image(LOGO_URI).props("width=512px ratio=1").classes("self-center").style("max-width: 80vw")
            with ui.column().classes("w-full items-center text-center gap-0 q-pb-md"):
                ui.icon("error", size="56px")
                ui.label("The app could not start").classes("text-h6 q-mt-sm")
                with ui.row().classes("gap-1 text-subtitle1"):
                    ui.label("Failed:")
                    ui.label(area).classes("text-weight-bold")
                if detail:
                    ui.markdown(detail).classes("aie-error-text text-body1 q-mt-md")
                ui.markdown(f"See the application log for details or refer to the [docs]({DOCS_URL}).").classes(
                    "aie-error-text text-body2 q-mt-md"
                )

    ui.run(host=HOST, port=PORT, title="AI Image Edit", dark=True, reload=False)
