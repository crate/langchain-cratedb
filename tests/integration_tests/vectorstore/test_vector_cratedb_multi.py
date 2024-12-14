import re

import pytest
import sqlalchemy as sa
from langchain_core.documents import Document

from langchain_cratedb import CrateDBVectorStore, CrateDBVectorStoreMultiCollection
from tests.integration_tests.vectorstore.fake_embeddings import (
    ConsistentFakeEmbeddingsWithAdaDimension,
    FakeEmbeddingsWithAdaDimension,
)
from tests.integration_tests.vectorstore.util import prune_document_ids


@pytest.mark.flaky(reruns=5)
def test_cratedb_multicollection_search_success(engine: sa.Engine) -> None:
    """
    `CrateDBVectorStoreMultiCollection` provides functionality for
    searching multiple collections.
    """

    store_1 = CrateDBVectorStore.from_texts(
        texts=["Räuber", "Hotzenplotz"],
        collection_name="test_collection_1",
        embedding=ConsistentFakeEmbeddingsWithAdaDimension(),
        connection=engine,
        pre_delete_collection=True,
    )
    _ = CrateDBVectorStore.from_texts(
        texts=["John", "Doe"],
        collection_name="test_collection_2",
        embedding=ConsistentFakeEmbeddingsWithAdaDimension(),
        connection=engine,
        pre_delete_collection=True,
    )

    # Probe the first store.
    output = store_1.similarity_search("Räuber", k=1)
    prune_document_ids(output)
    assert Document(page_content="Räuber") in output[:2]

    output = store_1.similarity_search("Hotzenplotz", k=2)
    prune_document_ids(output)
    assert Document(page_content="Hotzenplotz", metadata={}) in output[:2]

    output = store_1.similarity_search("John Doe", k=2)
    prune_document_ids(output)
    assert Document(page_content="Hotzenplotz", metadata={}) in output[:2]

    # Probe the multi-store.
    multisearch = CrateDBVectorStoreMultiCollection(
        collection_names=["test_collection_1", "test_collection_2"],
        embeddings=ConsistentFakeEmbeddingsWithAdaDimension(),
        connection=engine,
    )

    output = multisearch.similarity_search("Räuber Hotzenplotz", k=2)
    prune_document_ids(output)
    assert Document(page_content="Räuber") in output[:2]

    output = multisearch.similarity_search("John Doe", k=2)
    prune_document_ids(output)
    assert Document(page_content="John") in output[:2]


def test_cratedb_multicollection_fail_indexing_not_permitted(engine: sa.Engine) -> None:
    """
    `CrateDBVectorStoreMultiCollection` does not provide functionality for
    indexing documents.
    """

    with pytest.raises(NotImplementedError) as ex:
        CrateDBVectorStoreMultiCollection.from_texts(
            texts=["foo"],
            collection_names=["test_collection"],
            embedding=FakeEmbeddingsWithAdaDimension(),
            connection=engine,
        )
    assert ex.match(
        "The adapter for querying multiple collections "
        "can not be used for _indexing_ documents"
    )


def test_cratedb_multicollection_search_table_does_not_exist(engine: sa.Engine) -> None:
    """
    `CrateDBVectorStoreMultiCollection` will fail when the `collection`
    table does not exist.
    """

    store = CrateDBVectorStoreMultiCollection(
        collection_names=["unknown"],
        embeddings=ConsistentFakeEmbeddingsWithAdaDimension(),
        connection=engine,
    )
    with pytest.raises(sa.exc.ProgrammingError) as ex:
        store.similarity_search("foo")
    assert ex.match(
        re.escape("RelationUnknown[Relation 'langchain_collection' unknown]")
    )


def test_cratedb_multicollection_search_unknown_collection(engine: sa.Engine) -> None:
    """
    `CrateDBVectorStoreMultiCollection` will fail when not able to identify
    collections to search in.
    """

    CrateDBVectorStore.from_texts(
        texts=["Räuber", "Hotzenplotz"],
        collection_name="test_collection",
        embedding=ConsistentFakeEmbeddingsWithAdaDimension(),
        connection=engine,
        pre_delete_collection=True,
    )

    store = CrateDBVectorStoreMultiCollection(
        collection_names=["unknown"],
        embeddings=ConsistentFakeEmbeddingsWithAdaDimension(),
        connection=engine,
    )
    with pytest.raises(ValueError) as ex:
        store.similarity_search("foo")
    assert ex.match("No collections found")


def test_cratedb_multicollection_no_embedding_dimension(
    engine: sa.Engine, session: sa.orm.Session
) -> None:
    """
    Verify that addressing collections fails when not specifying dimensions.
    """
    store = CrateDBVectorStoreMultiCollection(
        embeddings=None,  # type: ignore[arg-type]
        connection=engine,
    )
    with pytest.raises(RuntimeError) as ex:
        store.get_collection(session)
    assert ex.match(
        "Collection can't be accessed without specifying "
        "dimension size of embedding vectors"
    )
