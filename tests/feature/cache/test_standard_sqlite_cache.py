"""
Test caching for LLMs and ChatModels, derived from tests for SQLite.

Source: https://github.com/langchain-ai/langchain/blob/langchain-core%3D%3D0.3.28/libs/community/tests/unit_tests/test_cache.py
"""

from typing import Dict, Generator, List, Union

import pytest
import sqlalchemy as sa
from _pytest.fixtures import FixtureRequest
from langchain_core.caches import BaseCache, InMemoryCache
from langchain_core.language_models import FakeListChatModel, FakeListLLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.load import dumps
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration
from sqlalchemy.orm import Session

from langchain_cratedb import CrateDBCache

try:
    from sqlalchemy.orm import declarative_base
except ImportError:
    from sqlalchemy.ext.declarative import declarative_base  # noqa: F401

from langchain.globals import get_llm_cache, set_llm_cache
from langchain_core.outputs import Generation


@pytest.fixture(params=["memory", "cratedb"])
def cache(request: FixtureRequest, engine: sa.Engine) -> BaseCache:
    if request.param == "memory":
        return InMemoryCache()
    if request.param == "cratedb":
        return CrateDBCache(engine=engine)
    raise NotImplementedError(f"Cache type not implemented: {request.param}")


@pytest.fixture(autouse=True)
def set_cache_and_teardown(cache: BaseCache) -> Generator[None, None, None]:
    # Will be run before each test
    set_llm_cache(cache)
    if llm_cache := get_llm_cache():
        llm_cache.clear()
    else:
        raise ValueError("Cache not set. This should never happen.")

    yield

    # Will be run after each test
    if llm_cache := get_llm_cache():
        llm_cache.clear()
        set_llm_cache(None)
    else:
        raise ValueError("Cache not set. This should never happen.")


async def test_llm_caching() -> None:
    prompt = "How are you?"
    response = "Test response"
    cached_response = "Cached test response"
    llm = FakeListLLM(responses=[response])
    if llm_cache := get_llm_cache():
        # sync test
        llm_cache.update(
            prompt=prompt,
            llm_string=create_llm_string(llm),
            return_val=[Generation(text=cached_response)],
        )
        assert llm.invoke(prompt) == cached_response
        # async test
        await llm_cache.aupdate(
            prompt=prompt,
            llm_string=create_llm_string(llm),
            return_val=[Generation(text=cached_response)],
        )
        assert await llm.ainvoke(prompt) == cached_response
    else:
        raise ValueError(
            "The cache not set. This should never happen, as the pytest fixture "
            "`set_cache_and_teardown` always sets the cache."
        )


def test_old_sqlite_llm_caching() -> None:
    llm_cache = get_llm_cache()
    if isinstance(llm_cache, CrateDBCache):
        prompt = "How are you?"
        response = "Test response"
        cached_response = "Cached test response"
        llm = FakeListLLM(responses=[response])
        items = [
            llm_cache.cache_schema(
                prompt=prompt,
                llm=create_llm_string(llm),
                response=cached_response,
                idx=0,
            )
        ]
        with Session(llm_cache.engine) as session, session.begin():
            for item in items:
                session.merge(item)
        assert llm.invoke(prompt) == cached_response


async def test_chat_model_caching() -> None:
    prompt: List[BaseMessage] = [HumanMessage(content="How are you?")]
    response = "Test response"
    cached_response = "Cached test response"
    cached_message = AIMessage(content=cached_response)
    llm = FakeListChatModel(responses=[response])
    if llm_cache := get_llm_cache():
        # sync test
        llm_cache.update(
            prompt=dumps(prompt),
            llm_string=llm._get_llm_string(),
            return_val=[ChatGeneration(message=cached_message)],
        )
        result = llm.invoke(prompt)
        assert isinstance(result, AIMessage)
        assert result.content == cached_response

        # async test
        await llm_cache.aupdate(
            prompt=dumps(prompt),
            llm_string=llm._get_llm_string(),
            return_val=[ChatGeneration(message=cached_message)],
        )
        result = await llm.ainvoke(prompt)
        assert isinstance(result, AIMessage)
        assert result.content == cached_response
    else:
        raise ValueError(
            "The cache not set. This should never happen, as the pytest fixture "
            "`set_cache_and_teardown` always sets the cache."
        )


async def test_chat_model_caching_params() -> None:
    prompt: List[BaseMessage] = [HumanMessage(content="How are you?")]
    response = "Test response"
    cached_response = "Cached test response"
    cached_message = AIMessage(content=cached_response)
    llm = FakeListChatModel(responses=[response])
    if llm_cache := get_llm_cache():
        # sync test
        llm_cache.update(
            prompt=dumps(prompt),
            llm_string=llm._get_llm_string(functions=[]),
            return_val=[ChatGeneration(message=cached_message)],
        )
        result = llm.invoke(prompt, functions=[])
        result_no_params = llm.invoke(prompt)
        assert isinstance(result, AIMessage)
        assert result.content == cached_response
        assert isinstance(result_no_params, AIMessage)
        assert result_no_params.content == response

        # async test
        await llm_cache.aupdate(
            prompt=dumps(prompt),
            llm_string=llm._get_llm_string(functions=[]),
            return_val=[ChatGeneration(message=cached_message)],
        )
        result = await llm.ainvoke(prompt, functions=[])
        result_no_params = await llm.ainvoke(prompt)
        assert isinstance(result, AIMessage)
        assert result.content == cached_response
        assert isinstance(result_no_params, AIMessage)
        assert result_no_params.content == response
    else:
        raise ValueError(
            "The cache not set. This should never happen, as the pytest fixture "
            "`set_cache_and_teardown` always sets the cache."
        )


async def test_llm_cache_clear() -> None:
    prompt = "How are you?"
    expected_response = "Test response"
    cached_response = "Cached test response"
    llm = FakeListLLM(responses=[expected_response])
    if llm_cache := get_llm_cache():
        # sync test
        llm_cache.update(
            prompt=prompt,
            llm_string=create_llm_string(llm),
            return_val=[Generation(text=cached_response)],
        )
        llm_cache.clear()
        response = llm.invoke(prompt)
        assert response == expected_response

        # async test
        await llm_cache.aupdate(
            prompt=prompt,
            llm_string=create_llm_string(llm),
            return_val=[Generation(text=cached_response)],
        )
        await llm_cache.aclear()
        response = await llm.ainvoke(prompt)
        assert response == expected_response
    else:
        raise ValueError(
            "The cache not set. This should never happen, as the pytest fixture "
            "`set_cache_and_teardown` always sets the cache."
        )


def create_llm_string(llm: Union[BaseLLM, BaseChatModel]) -> str:
    _dict: Dict = llm.dict()
    _dict["stop"] = None
    return str(sorted(_dict.items()))
