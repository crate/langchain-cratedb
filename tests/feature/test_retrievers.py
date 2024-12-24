from typing import Type

import pytest
from langchain_tests.integration_tests import (
    RetrieversIntegrationTests,
)

from langchain_cratedb.retrievers import CrateDBRetriever


@pytest.mark.skip("CrateDBRetriever not implemented yet")
class TestCrateDBRetriever(RetrieversIntegrationTests):
    @property
    def retriever_constructor(self) -> Type[CrateDBRetriever]:
        """Get an empty vectorstore for unit tests."""
        return CrateDBRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        return {"k": 2}

    @property
    def retriever_query_example(self) -> str:
        """
        Returns a str representing the "query" of an example retriever call.
        """
        return "example query"
