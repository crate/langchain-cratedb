"""
Test standard and semantic caching.
Derived from Memcached and SQLAlchemy.

Source: https://github.com/langchain-ai/langchain/blob/langchain-core%3D%3D0.3.28/libs/community/tests/integration_tests/cache/test_memcached_cache.py
"""

import pytest
import sqlalchemy as sa
from langchain_core.caches import BaseCache
from langchain_core.globals import set_llm_cache
from langchain_core.outputs import Generation, LLMResult

from langchain_cratedb import CrateDBCache
from tests.utils import FakeLLM, get_llm_cache


@pytest.fixture()
def cache(engine: sa.Engine) -> BaseCache:
    return CrateDBCache(engine=engine)


def test_memcached_cache(cache: BaseCache) -> None:
    """Test general caching"""

    set_llm_cache(cache)
    llm = FakeLLM()

    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted(params.items()))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])
    output = llm.generate(["foo"])
    expected_output = LLMResult(
        generations=[[Generation(text="fizz")]],
        llm_output={},
    )
    assert output == expected_output
    # clear the cache
    get_llm_cache().clear()


def test_memcached_cache_flush(cache: BaseCache) -> None:
    """Test flushing cache"""

    set_llm_cache(cache)
    llm = FakeLLM()

    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted(params.items()))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])
    output = llm.generate(["foo"])
    expected_output = LLMResult(
        generations=[[Generation(text="fizz")]],
        llm_output={},
    )
    assert output == expected_output
    # clear the cache
    get_llm_cache().clear(delay=0, noreply=False)

    # After cache has been cleared, the result shouldn't be the same
    output = llm.generate(["foo"])
    assert output != expected_output


def test_sqlalchemy_cache(engine: sa.Engine) -> None:
    """Test custom_caching behavior."""

    from sqlalchemy_cratedb.support import patch_autoincrement_timestamp

    patch_autoincrement_timestamp()

    Base = sa.orm.declarative_base()

    class FulltextLLMCache(Base):  # type: ignore
        """CrateDB table for fulltext-indexed LLM Cache."""

        __tablename__ = "llm_cache_fulltext"
        # TODO: Original. Can it be converged by adding a polyfill to
        #       `sqlalchemy-cratedb`?
        # id = Column(Integer, Sequence("cache_id"), primary_key=True)  # noqa: ERA001
        id = sa.Column(sa.BigInteger, server_default=sa.func.now(), primary_key=True)
        prompt = sa.Column(sa.String, nullable=False)
        llm = sa.Column(sa.String, nullable=False)
        idx = sa.Column(sa.Integer)
        response = sa.Column(sa.String)

    set_llm_cache(CrateDBCache(engine, FulltextLLMCache))
    get_llm_cache().clear()

    llm = FakeLLM()
    params = llm.dict()
    params["stop"] = None
    llm_string = str(sorted(params.items()))
    get_llm_cache().update("foo", llm_string, [Generation(text="fizz")])
    output = llm.generate(["foo", "bar", "foo"])
    expected_cache_output = [Generation(text="foo")]
    cache_output = get_llm_cache().lookup("bar", llm_string)
    assert cache_output == expected_cache_output
    set_llm_cache(None)
    expected_generations = [
        [Generation(text="fizz")],
        [Generation(text="foo")],
        [Generation(text="fizz")],
    ]
    expected_output = LLMResult(
        generations=expected_generations,
        llm_output=None,
    )
    assert output == expected_output
