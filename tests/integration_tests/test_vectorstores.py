from typing import AsyncGenerator, Generator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import (
    AsyncReadWriteTestSuite,
    ReadWriteTestSuite,
)

from langchain_cratedb.vectorstores import CrateDBVectorStore


class TestCrateDBVectorStoreSync(ReadWriteTestSuite):
    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        store = CrateDBVectorStore()
        # note: store should be EMPTY at this point
        # if you need to delete data, you may do so here
        try:
            yield store
        finally:
            # cleanup operations, or deleting data
            pass


class TestCrateDBVectorStoreAsync(AsyncReadWriteTestSuite):
    @pytest.fixture()
    async def vectorstore(self) -> AsyncGenerator[VectorStore, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        store = CrateDBVectorStore()
        # note: store should be EMPTY at this point
        # if you need to delete data, you may do so here
        try:
            yield store
        finally:
            # cleanup operations, or deleting data
            pass
