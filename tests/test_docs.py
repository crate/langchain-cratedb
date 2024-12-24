from pathlib import Path

import nbclient
import pytest
from _pytest.python import Metafunc

from tests.util.notebook import generate_notebook_tests, list_notebooks, run_notebook

ROOT = Path(__file__).parent.parent
NOTEBOOKS_FOLDER = ROOT / "docs"


# Configure notebooks to skip testing.
# TODO: Enable testing for all notebooks.
SKIP_NOTEBOOKS = [
    "loaders.ipynb",
    "retrievers.ipynb",
    "stores.ipynb",
    "vectorstores.ipynb",
]


def pytest_generate_tests(metafunc: Metafunc) -> None:
    """
    Generate pytest test case per Jupyter Notebook.
    """
    notebooks_root = NOTEBOOKS_FOLDER
    notebook_paths = list_notebooks(notebooks_root)
    generate_notebook_tests(metafunc, notebook_paths=notebook_paths)


def test_notebook(notebook: Path) -> None:
    """
    Execute Jupyter Notebook, one test case per .ipynb file.

    Skip test cases that trip when no OpenAI API key is configured.
    """
    if notebook.name in SKIP_NOTEBOOKS:
        raise pytest.skip(f"FIXME: Excluding notebook: {notebook.name}")
    try:
        run_notebook(notebook)
    except nbclient.exceptions.CellExecutionError as ex:
        if "The api_key client option must be set" not in str(ex):
            raise
        raise pytest.skip(
            "Skipping test because `OPENAI_API_KEY` is not defined"
        ) from ex
