import typing as t
from pathlib import Path

import openai
import pytest
from _pytest.capture import CaptureFixture
from _pytest.python import Metafunc

from tests.util.python import generate_file_tests, list_python_files

ROOT = Path(__file__).parent.parent
EXAMPLES_FOLDER = ROOT / "examples"


# Configure example programs to skip testing.
SKIP_FILES: t.List[str] = []


def test_dummy(run_file: t.Callable, capsys: CaptureFixture) -> None:
    run_file(EXAMPLES_FOLDER / "dummy.py")
    out, err = capsys.readouterr()
    assert out == "Hallo, Räuber Hotzenplotz.\n"


def pytest_generate_tests(metafunc: Metafunc) -> None:
    """
    Generate pytest test case per example program.
    """
    examples_root = EXAMPLES_FOLDER
    file_paths = list_python_files(examples_root)
    generate_file_tests(metafunc, file_paths=file_paths)


def test_file(run_file: t.Callable, file: Path) -> None:
    """
    Execute Python code, one test case per .py file.

    Skip test cases that trip when no OpenAI API key is configured.
    """
    if file.name in SKIP_FILES:
        raise pytest.skip(f"FIXME: Skipping file: {file.name}")
    try:
        run_file(file)
    except openai.OpenAIError as ex:
        if "The api_key client option must be set" not in str(ex):
            raise
        raise pytest.skip(
            "Skipping test because `OPENAI_API_KEY` is not defined"
        ) from ex
