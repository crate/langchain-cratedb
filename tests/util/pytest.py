# TODO: Vendored from `pueblo.testing.snippet`.
#       Reason: It trips with an import error after just installing it.
#       ImportError: cannot import name 'FixtureDef' from 'pytest'
#       https://github.com/pyveci/pueblo/issues/129
from __future__ import annotations

import importlib
import typing as t
from os import PathLike
from pathlib import Path

import pytest


def run_module_function(
    request: pytest.FixtureRequest,
    filepath: t.Union[str, Path],
    entrypoint: str = "main",
) -> tuple[PathLike[str] | str, int | None, str]:
    from _pytest.monkeypatch import MonkeyPatch
    from _pytest.python import Function

    path = Path(filepath)

    # Temporarily add parent directory to module search path.
    with MonkeyPatch.context() as m:
        m.syspath_prepend(path.parent)

        # Import file as Python module.
        try:
            mod = importlib.import_module(path.stem)
        except ImportError as ex:
            raise ImportError(f"Module not found at {filepath}: {ex}") from ex
        fun = getattr(mod, entrypoint)

        # Wrap the entrypoint function into a pytest test case, and run it.
        test = Function.from_parent(request.node, name=entrypoint, callobj=fun)
        test.runtest()
        return test.reportinfo()
