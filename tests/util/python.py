# TODO: Vendored from `pueblo.testing.snippet`.
#       Reason: It trips with an import error after just installing it.
#       ImportError: cannot import name 'FixtureDef' from 'pytest'
#       https://github.com/pyveci/pueblo/issues/129
import typing as t
from os import PathLike
from pathlib import Path

import pytest
from _pytest.fixtures import FixtureRequest
from _pytest.python import Metafunc

from tests.util.pytest import run_module_function


def list_python_files(path: Path) -> t.List[Path]:
    """
    Enumerate all Python files found in given directory, recursively.
    """
    return list(path.rglob("*.py"))


def generate_file_tests(
    metafunc: Metafunc, file_paths: t.List[Path], fixture_name: str = "file"
) -> None:
    """
    Generate test cases for Python example programs.
    """
    if fixture_name in metafunc.fixturenames:
        names = [nb_path.name for nb_path in file_paths]
        metafunc.parametrize(fixture_name, file_paths, ids=names)


@pytest.fixture
def run_file(request: FixtureRequest) -> t.Callable:
    """
    Invoke Python example programs as pytest test cases. This fixture is a factory
    that returns a function that can be used for invocation.

    TODO: Wrap outcome into a better shape.
          At least, use a dictionary, optimally an object.
    """

    def _runner(
        path: Path,
    ) -> t.Tuple[t.Union[PathLike[str], str], t.Union[int, None], str]:
        outcome = run_module_function(request=request, filepath=path)
        assert isinstance(outcome[0], Path)
        return outcome

    return _runner
