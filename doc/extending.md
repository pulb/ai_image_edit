# Adding a model or frontend

Adding a model means implementing `ModelBackend` (see `src/ai_image_edit/models/base.py`) and
registering one loader in `src/ai_image_edit/models/__init__.py`.

Adding a frontend means writing a module that exposes
`run(model: ModelBackend) -> None` and registering it in
`src/ai_image_edit/frontends/__init__.py`.

Add the new backend's or frontend's dependencies as an extra in
`pyproject.toml`, named `<name>_backend` or `<name>_frontend`.
