from typing import Type

from langchain_tests.integration_tests import (
    RetrieversIntegrationTests,
)

from langchain_cratedb.retrievers import CrateDBRetriever


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
        Returns a dictionary representing the "args" of an example retriever call.
        """
        return "example query"
