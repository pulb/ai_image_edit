# Adding a model or frontend

Adding a model means implementing `ModelBackend` (see `models/base.py`) and
registering one loader in `models/__init__.py`.

Adding a frontend means writing a module that exposes
`run(model: ModelBackend, model_backend: str) -> None` and registering it in
`frontends/__init__.py`.
