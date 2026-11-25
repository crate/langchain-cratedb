import os
import re
from pathlib import Path

import openai
import pytest

ROOT = Path(__file__).parent.parent
SOURCES = [*(ROOT / "examples").rglob("*.py"), ROOT / "docs" / "cache.ipynb"]
MODEL_NAME = re.compile(r"\b(?:gpt|text-embedding)-[\w.-]*\w")


def model_names() -> list[str]:
    return sorted(
        {name for path in SOURCES for name in MODEL_NAME.findall(path.read_text())}
    )


@pytest.mark.parametrize("name", model_names())
def test_openai_model_exists(name: str) -> None:
    """
    Every OpenAI model named by an example program or the cache notebook is offered.
    """
    if "OPENAI_API_KEY" not in os.environ:
        pytest.skip("Skipping test because `OPENAI_API_KEY` is not defined")
    try:
        openai.OpenAI().models.retrieve(name)
    except openai.OpenAIError as ex:
        if "no credits remaining" in str(ex):
            pytest.skip("Skipping test because the OpenAI account has no credits")
        raise
