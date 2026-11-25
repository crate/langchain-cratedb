import sys
import typing as t
from pathlib import Path

import langchain_openai
import pytest
from _pytest.capture import CaptureFixture
from _pytest.python import Metafunc
from langchain_core.language_models import FakeListChatModel

from tests.feature.vectorstore.fake_embeddings import (
    ConsistentFakeEmbeddingsWithAdaDimension,
)
from tests.util.python import generate_file_tests, list_python_files

ROOT = Path(__file__).parent.parent
EXAMPLES_FOLDER = ROOT / "examples"


# Configure example programs to skip testing.
SKIP_FILES: t.List[str] = []


@pytest.fixture
def scripted_openai(monkeypatch: pytest.MonkeyPatch, file: Path) -> t.Iterator[None]:
    """
    Replace the OpenAI chat model and embeddings by offline substitutes.

    Example programs bind both names when they are imported, so the program is
    imported afresh while the substitutes are in place, and dropped afterwards.
    """
    monkeypatch.setattr(
        langchain_openai,
        "ChatOpenAI",
        lambda **kwargs: FakeListChatModel(responses=["The answer is 42."], **kwargs),
    )
    monkeypatch.setattr(
        langchain_openai, "OpenAIEmbeddings", ConsistentFakeEmbeddingsWithAdaDimension
    )
    sys.modules.pop(file.stem, None)
    yield
    sys.modules.pop(file.stem, None)


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


def test_file(run_file: t.Callable, file: Path, scripted_openai: None) -> None:
    """
    Execute Python code, one test case per .py file.

    Skip test cases that need a reachable Ollama server.
    """
    if file.name in SKIP_FILES:
        raise pytest.skip(f"FIXME: Skipping file: {file.name}")
    try:
        run_file(file)
    except ConnectionError as ex:
        if "Failed to connect to Ollama" not in str(ex):
            raise
        raise pytest.skip("Skipping test because Ollama is not reachable") from ex
