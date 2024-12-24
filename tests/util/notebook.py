# TODO: Vendored from `pueblo.testing.notebook`.
#       Reason: It trips with an import error after just installing it.
#       ImportError: cannot import name 'FixtureDef' from 'pytest'
#       https://github.com/pyveci/pueblo/issues/129
import typing as t
from pathlib import Path

import pytest
from _pytest.python import Metafunc


def list_notebooks(path: Path) -> t.List[Path]:
    """
    Enumerate all Jupyter Notebook files found in given directory.
    """
    return list(path.rglob("*.ipynb"))


def generate_notebook_tests(
    metafunc: Metafunc, notebook_paths: t.List[Path], fixture_name: str = "notebook"
) -> None:
    """
    Generate test cases for Jupyter Notebooks.
    To be used from `pytest_generate_tests`.
    """
    if fixture_name in metafunc.fixturenames:
        names = [nb_path.name for nb_path in notebook_paths]
        metafunc.parametrize(fixture_name, notebook_paths, ids=names)


def run_notebook(
    notebook: Path,
    enable_skipping: bool = True,
    timeout: float = 60,
    **kwargs: t.Dict[str, t.Any],
) -> None:
    """
    Execute Jupyter Notebook, one test case per .ipynb file, with optional skipping.

    Skip executing a notebook by using this code within a cell::

        pytest.exit("Something failed but let's skip! [skip-notebook]")

    For example, this is used by `pueblo.util.environ.getenvpass()`, to
    skip executing the notebook when an authentication token is not supplied.
    """

    from nbclient.exceptions import CellExecutionError
    from testbook import testbook

    with testbook(notebook, timeout=timeout, **kwargs) as tb:
        try:
            tb.execute()

        # Skip notebook if `pytest.exit()` is invoked,
        # including the `[skip-notebook]` label.
        except CellExecutionError as ex:
            if enable_skipping:
                msg = str(ex)
                if "[skip-notebook]" in msg:
                    raise pytest.skip(msg) from ex
            raise
