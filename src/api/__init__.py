"""HTTP surface package.

`app` is resolved lazily so that importing a low-level module (the LLM provider,
for example) does not drag in FastAPI, the pipeline, or its storage.
"""

__all__ = ["app"]


def __getattr__(name):
    if name == "app":
        from src.api.server import app as fastapi_app

        return fastapi_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
