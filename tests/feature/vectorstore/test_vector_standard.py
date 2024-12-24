from typing import Generator

import pytest
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests

from langchain_cratedb.vectorstores import CrateDBVectorStore


class TestCrateDBVectorStore(VectorStoreIntegrationTests):
    @property
    def has_async(self) -> bool:
        return False

    @pytest.fixture()
    def vectorstore(self, engine) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        store = CrateDBVectorStore(
            self.get_embeddings(),
            connection=engine,
            engine_args={"echo": True},
        )
        # note: store should be EMPTY at this point
        # if you need to delete data, you may do so here
        try:
            yield store
        finally:
            # cleanup operations, or deleting data
            pass

    @pytest.mark.xfail(
        reason=(
            "CrateDB: Like `langchain-postgres`, this adapter raises "
            '`ValueError("Collection not found")` when accessing a '
            "non-existing collection."
        )
    )
    def test_vectorstore_is_empty(self, vectorstore: VectorStore) -> None:
        """Test that the vectorstore is empty.

        TODO: This and the next test case method need to be overwritten, because
              CrateDB's test harness starts with an empty database. This is
              different with `langchain-postgres`, as it provisions the database
              schema when setting up the standard tests already, and, by doing so,
              are masking the constraint that accessing an empty or non-existing
              collection should not raise an error, as this test case suggests.
              .
              On the other hand, the documentation of this test case says:
              .
                If this test fails, check that the test class (i.e., sub class of
                ``VectorStoreIntegrationTests``) initializes an empty vector store
                in the ``vectorestore`` fixture.
              .
              So, we are wondering what exactly this test case aims to test:
              .
              1. Is it validating that LangChain should not trip at runtime when
                 accessing a collection that does not exist?
              2. Is it merely validating that the test suite setup provisions the
                 database schema correctly?
              .
              If 1. is the case, we are asking for clarification. If 2. is the case,
              we think we should add corresponding setup code to this class,
              like `langchain-postgres` is doing it.
        """
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        try:
            assert vectorstore.similarity_search("foo", k=1) == []
        except ValueError as ex:
            if str(ex) != "Collection not found":
                raise

    @pytest.mark.xfail(
        reason=(
            "CrateDB: Like `langchain-postgres`, this adapter raises "
            '`ValueError("Collection not found")` when accessing a '
            "non-existing collection."
        )
    )
    def test_vectorstore_still_empty(self, vectorstore: VectorStore) -> None:
        """This test should follow a test that adds documents.

        This verifies that the fixture is set up properly to be empty
        after each test, i.e. that this class properly clears the vector
        store in the ``finally`` block.
        """
        if not self.has_sync:
            pytest.skip("Sync tests not supported.")

        try:
            assert vectorstore.similarity_search("foo", k=1) == []
        except ValueError as ex:
            if str(ex) != "Collection not found":
                raise

    # ``get_by_ids`` was added to the ``VectorStore`` interface in
    # ``langchain-core`` version 0.2.11. If difficult to implement, this
    # test can be skipped using a pytest ``xfail`` on the test class.

    @pytest.mark.xfail(
        reason=(
            "CrateDB: Write order does not necessarily reflect read order. "
            "This means `get_by_ids` is not stable."
        )
    )
    def test_add_documents_documents(self, vectorstore: VectorStore) -> None:
        super().test_add_documents_documents(vectorstore)

    @pytest.mark.xfail(
        reason=(
            "CrateDB: Write order does not necessarily reflect read order. "
            "This means `get_by_ids` is not stable."
        )
    )
    def test_add_documents_with_existing_ids(self, vectorstore: VectorStore) -> None:
        super().test_add_documents_with_existing_ids(vectorstore)
