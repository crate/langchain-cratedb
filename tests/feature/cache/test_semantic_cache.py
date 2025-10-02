"""
Test semantic cache.
Derived from SingleStoreDB.

Source: https://github.com/langchain-ai/langchain/blob/langchain-core%3D%3D0.3.28/libs/community/tests/integration_tests/cache/test_singlestoredb_cache.py
"""

import typing as t
import uuid

import pytest
import sqlalchemy as sa
from langchain.globals import get_llm_cache, set_llm_cache
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.load import dumps
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult

from langchain_cratedb import CrateDBSemanticCache
from tests.feature.cache.fake_embeddings import (
    ConsistentFakeEmbeddings,
    FakeEmbeddings,
)
from tests.utils import FakeLLM


def random_string() -> str:
    return str(uuid.uuid4())


@pytest.fixture(autouse=True)
def set_cache_and_teardown() -> t.Generator[None, None, None]:
    yield
    set_llm_cache(None)


def test_semantic_cache_single(engine: sa.Engine) -> None:
    """
    Test semantic cache functionality with single item.
    Derived from OpenSearch/SingleStore.
    """
    set_llm_cache(
        CrateDBSemanticCache(
            embedding=FakeEmbeddings(),
            connection=engine,
            search_threshold=1.0,
        )
    )
    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted(params.items()))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz")]

    get_llm_cache().clear(llm_string=llm_string)
    output = get_llm_cache().lookup("bar", llm_string)
    assert output != [Generation(text="fizz")]


def test_semantic_cache_multi(engine: sa.Engine) -> None:
    """
    Test semantic cache functionality with multiple items.
    Derived from OpenSearch/SingleStore.
    """
    set_llm_cache(
        CrateDBSemanticCache(
            embedding=FakeEmbeddings(),
            connection=engine,
            search_threshold=1.0,
        )
    )

    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted(params.items()))
    get_llm_cache().update(
        "foo", llm_string, [Generation(text="fizz"), Generation(text="Buzz")]
    )

    # foo and bar will have the same embedding produced by FakeEmbeddings
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == [Generation(text="fizz"), Generation(text="Buzz")]

    # clear the cache
    get_llm_cache().clear(llm_string=llm_string)
    output = get_llm_cache().lookup("bar", llm_string)
    assert output != [Generation(text="fizz"), Generation(text="Buzz")]


def test_semantic_cache_chat(engine: sa.Engine) -> None:
    """
    Test semantic cache functionality for chat messages.
    Derived from Redis.
    """
    set_llm_cache(
        CrateDBSemanticCache(
            embedding=FakeEmbeddings(),
            connection=engine,
            search_threshold=1.0,
        )
    )
    llm = FakeChatModel()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted(params.items()))
    prompt: t.List[BaseMessage] = [HumanMessage(content="foo")]
    llm_cache = t.cast(CrateDBSemanticCache, get_llm_cache())
    llm_cache.update(
        dumps(prompt), llm_string, [ChatGeneration(message=AIMessage(content="fizz"))]
    )
    output = llm.generate([prompt])
    expected_output = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="fizz"))]],
        llm_output={},
    )

    # Prune metadata information.
    output.run = None
    delattr(output.generations[0][0].message, "usage_metadata")  # type: ignore[union-attr]
    delattr(expected_output.generations[0][0].message, "usage_metadata")  # type: ignore[union-attr]

    assert output == expected_output
    llm_cache.clear(llm_string=llm_string)


@pytest.mark.parametrize("embedding", [ConsistentFakeEmbeddings()])
@pytest.mark.parametrize(
    "prompts,  generations",
    [
        # Single prompt, single generation
        ([random_string()], [[random_string()]]),
        # Single prompt, multiple generations
        ([random_string()], [[random_string(), random_string()]]),
        # Single prompt, multiple generations
        ([random_string()], [[random_string(), random_string(), random_string()]]),
        # ruff: noqa: ERA001
        # Multiple prompts, multiple generations
        # (
        #    [random_string(), random_string()],
        #    [[random_string()], [random_string(), random_string()]],
        # ),
    ],
    ids=[
        "single_prompt_single_generation",
        "single_prompt_multiple_generations",
        "single_prompt_multiple_generations",
        # "multiple_prompts_multiple_generations",
    ],
)
def test_semantic_cache_hit(
    embedding: Embeddings,
    prompts: t.List[str],
    generations: t.List[t.List[str]],
    engine: sa.Engine,
) -> None:
    """
    Test semantic cache functionality with hits.
    Derived from Redis.
    """
    set_llm_cache(
        CrateDBSemanticCache(
            embedding=FakeEmbeddings(),
            connection=engine,
            search_threshold=1.0,
        )
    )

    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted(params.items()))

    llm_generations = [
        [
            Generation(text=generation, generation_info=params)
            for generation in prompt_i_generations
        ]
        for prompt_i_generations in generations
    ]
    llm_cache = t.cast(CrateDBSemanticCache, get_llm_cache())
    for prompt_i, llm_generations_i in zip(prompts, llm_generations):
        print(prompt_i)  # noqa: T201
        print(llm_generations_i)  # noqa: T201
        llm_cache.update(prompt_i, llm_string, llm_generations_i)
    llm.generate(prompts)
    assert llm.generate(prompts) == LLMResult(
        generations=llm_generations, llm_output={}
    )
