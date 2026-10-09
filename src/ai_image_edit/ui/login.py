# SPDX-License-Identifier: GPL-3.0-or-later
"""Password login for the UI."""
import asyncio
import hmac
from typing import Optional
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import RedirectResponse
from nicegui import app, ui
from starlette.middleware.base import BaseHTTPMiddleware

from ai_image_edit.ui.components import apply_dark_theme
from ai_image_edit.ui.helpers import storage_secret


def install_password_login(password: str) -> str:
    """
    Puts every route behind a password login page and returns the
    storage_secret ui.run() needs for the session cookie.

    Only /login, /favicon.ico and NiceGUI's own /_nicegui assets are open;
    /files (uploaded and generated images) is covered like any other route.
    The secret is derived from the password unless APP_STORAGE_SECRET is
    set, so logins survive restarts and a changed password logs everyone out.
    """
    open_routes = {"/favicon.ico", "/login"}

    @app.add_middleware
    class AuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            path = request.url.path
            if app.storage.user.get("authenticated") or path in open_routes or path.startswith("/_nicegui"):
                return await call_next(request)
            return RedirectResponse(f"/login?redirect_to={quote(path)}")

    @ui.page("/login")
    def login_page(redirect_to: str = "/") -> Optional[RedirectResponse]:
        if app.storage.user.get("authenticated"):
            return RedirectResponse("/")

        # Only same-site paths: "//host" and "/\host" would redirect off-site.
        target = redirect_to if redirect_to.startswith("/") and not redirect_to.startswith("//") and "\\" not in redirect_to else "/"

        async def try_login() -> None:
            if hmac.compare_digest(field.value.encode(), password.encode()):
                app.storage.user["authenticated"] = True
                ui.navigate.to(target)
            else:
                await asyncio.sleep(1.0)
                ui.notify("Wrong password", color="negative")

        apply_dark_theme()
        with ui.card().classes("absolute-center items-stretch"):
            field = ui.input("Password", password=True, password_toggle_button=True).props("autofocus")
            field.on("keydown.enter", try_login)
            ui.button("Log in", on_click=try_login)
        return None

    return storage_secret(password)
