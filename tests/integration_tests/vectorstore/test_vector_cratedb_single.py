"""
Validate LangChain using CrateDB's `FLOAT_VECTOR` / `KNN_MATCH` functionality.
"""

import contextlib
from typing import Any, Dict, Generator, List, Optional, Sequence, cast

import pytest
import sqlalchemy as sa
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_cratedb.vectorstores import (
    CrateDBVectorStore,
)
from tests.integration_tests.conftest import CONNECTION_STRING
from tests.fixtures.filtering_test_cases import (
    DOCUMENTS,
    TYPE_1_FILTERING_TEST_CASES,
    TYPE_2_FILTERING_TEST_CASES,
    TYPE_3_FILTERING_TEST_CASES,
    TYPE_4_FILTERING_TEST_CASES,
    TYPE_5_FILTERING_TEST_CASES,
    TYPE_6_FILTERING_TEST_CASES,
)
from tests.integration_tests.vectorstore.fake_embeddings import (
    ADA_TOKEN_COUNT,
    ConsistentFakeEmbeddingsWithAdaDimension,
    FakeEmbeddingsWithAdaDimension,
)
from tests.integration_tests.vectorstore.util import (
    ensure_collection,
    prune_document_ids,
)


def test_cratedb_collection_read_only(session: sa.orm.Session) -> None:
    """
    Test using a collection, without adding any embeddings upfront.

    This happens when just invoking the "retrieval" case.

    In this scenario, embedding dimensionality needs to be figured out
    from the supplied `embedding_function`.
    """

    # Create a fake collection item.
    ensure_collection(session, "baz2")

    # This test case needs an embedding _with_ dimensionality.
    # Otherwise, the data access layer is unable to figure it
    # out at runtime.
    embedding = ConsistentFakeEmbeddingsWithAdaDimension()

    vectorstore = CrateDBVectorStore(
        collection_name="baz2",
        connection=CONNECTION_STRING,
        embeddings=embedding,
    )
    output = vectorstore.similarity_search("foo", k=1)

    # No documents/embeddings have been loaded, the collection is empty.
    # This is why there are also no results.
    assert output == []


def test_cratedb_texts(engine: sa.Engine) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection=engine,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    prune_document_ids(output)
    assert output == [Document(metadata={}, page_content="foo")]


def test_cratedb_embedding_dimension(engine: sa.Engine) -> None:
    """Verify the `embedding` column uses the correct vector dimensionality."""
    texts = ["foo", "bar", "baz"]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=ConsistentFakeEmbeddingsWithAdaDimension(),
        connection=engine,
        pre_delete_collection=True,
    )
    with docsearch._make_sync_session() as session:
        result = session.execute(sa.text("SHOW CREATE TABLE langchain_embedding"))
        record = result.first()
        if not record:
            raise ValueError("No data found")
        ddl = record[0]
        assert f'"embedding" FLOAT_VECTOR({ADA_TOKEN_COUNT})' in ddl


def test_cratedb_embeddings(engine: sa.Engine) -> None:
    """Test end to end construction with embeddings and search."""
    texts = ["foo", "bar", "baz"]
    text_embeddings = FakeEmbeddingsWithAdaDimension().embed_documents(texts)
    text_embedding_pairs = list(zip(texts, text_embeddings))
    docsearch = CrateDBVectorStore.from_embeddings(
        text_embeddings=text_embedding_pairs,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection=engine,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    prune_document_ids(output)
    assert output == [Document(page_content="foo")]


def test_cratedb_with_metadatas(engine: sa.Engine) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=engine,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search("foo", k=1)
    prune_document_ids(output)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


def test_cratedb_with_metadatas_with_scores(engine: sa.Engine) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=engine,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score("foo", k=1)
    prune_document_ids(output)
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 1.0)]


def test_cratedb_with_filter_match(engine: sa.Engine) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=engine,
        pre_delete_collection=True,
    )
    # TODO: Original:
    #       assert output == [(Document(page_content="foo", metadata={"page": "0"}), 0.0)]  # noqa: E501
    output = docsearch.similarity_search_with_score("foo", k=1, filter={"page": "0"})
    prune_document_ids(output)
    assert output == [(Document(page_content="foo", metadata={"page": "0"}), 1.0)]


def test_cratedb_with_filter_distant_match(engine: sa.Engine) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=engine,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score("foo", k=2, filter={"page": "2"})
    prune_document_ids(output)
    # Original score value: 0.0013003906671379406
    assert output == [(Document(page_content="baz", metadata={"page": "2"}), 0.2)]


def test_cratedb_with_filter_no_match(engine: sa.Engine) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=engine,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score("foo", k=1, filter={"page": "5"})
    assert output == []


def test_cratedb_collection_delete(engine: sa.Engine, session: sa.orm.Session) -> None:
    """
    Test end to end collection construction and deletion.
    Uses two different collections of embeddings.
    """

    store_foo = CrateDBVectorStore.from_texts(
        texts=["foo"],
        collection_name="test_collection_foo",
        collection_metadata={"category": "foo"},
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=[{"document": "foo"}],
        connection=engine,
        pre_delete_collection=True,
    )
    store_bar = CrateDBVectorStore.from_texts(
        texts=["bar"],
        collection_name="test_collection_bar",
        collection_metadata={"category": "bar"},
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=[{"document": "bar"}],
        connection=engine,
        pre_delete_collection=True,
    )

    # Verify data in database.
    collection_foo = store_foo.get_collection(session)
    collection_bar = store_bar.get_collection(session)
    if collection_foo is None or collection_bar is None:
        assert False, "Expected CollectionStore objects but received None"
    assert collection_foo.embeddings[0].cmetadata == {"document": "foo"}
    assert collection_bar.embeddings[0].cmetadata == {"document": "bar"}

    # Delete first collection.
    store_foo.delete_collection()

    # Verify that the "foo" collection has been deleted.
    collection_foo = store_foo.get_collection(session)
    collection_bar = store_bar.get_collection(session)
    if collection_bar is None:
        assert False, "Expected CollectionStore object but received None"
    assert collection_foo is None
    assert collection_bar.embeddings[0].cmetadata == {"document": "bar"}

    # Verify that associated embeddings also have been deleted.
    embeddings_count = session.query(store_foo.EmbeddingStore).count()
    assert embeddings_count == 1


def test_cratedb_collection_with_metadata(
    engine: sa.Engine, session: sa.orm.Session
) -> None:
    """Test end to end collection construction"""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    cratedb_vector = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection",
        collection_metadata={"foo": "bar"},
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=engine,
        pre_delete_collection=True,
    )
    collection = cratedb_vector.get_collection(session)
    if collection is None:
        assert False, "Expected a CollectionStore object but received None"
    else:
        assert collection.name == "test_collection"
        assert collection.cmetadata == {"foo": "bar"}


def test_cratedb_collection_no_embedding_dimension(
    engine: sa.Engine, session: sa.orm.Session
) -> None:
    """
    Verify that addressing collections fails when not specifying dimensions.
    """
    cratedb_vector = CrateDBVectorStore(
        embeddings=None,  # type: ignore[arg-type]
        connection=engine,
    )
    with pytest.raises(RuntimeError) as ex:
        cratedb_vector.get_collection(session)
    assert ex.match(
        "Collection can't be accessed without specifying "
        "dimension size of embedding vectors"
    )


@pytest.mark.parametrize("operator", ["$in", "IN"])
def test_cratedb_with_filter_in_set(engine: sa.Engine, operator: str) -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=engine,
        pre_delete_collection=True,
    )
    output = docsearch.similarity_search_with_score(
        "foo", k=2, filter={"page": {operator: ["0", "2"]}}
    )
    prune_document_ids(output)
    # Original score values: 0.0, 0.0013003906671379406
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0),
        (Document(page_content="baz", metadata={"page": "2"}), 0.2),
    ]


def test_cratedb_delete_docs(engine: sa.Engine) -> None:
    """Add and delete documents."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection_filter",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        ids=["1", "2", "3"],
        connection=engine,
        pre_delete_collection=True,
    )
    docsearch.delete(["1", "2"])
    with docsearch._make_sync_session() as session:
        records = list(session.query(docsearch.EmbeddingStore).all())
        # ignoring type error since mypy cannot determine whether
        # the list is sortable
        assert sorted(record.id for record in records) == ["3"]  # type: ignore

    docsearch.delete(["2", "3"])  # Should not raise on missing ids
    with docsearch._make_sync_session() as session:
        records = list(session.query(docsearch.EmbeddingStore).all())
        # ignoring type error since mypy cannot determine whether
        # the list is sortable
        assert sorted(record.id for record in records) == []  # type: ignore


def test_cratedb_relevance_score(engine: sa.Engine) -> None:
    """Test to make sure the relevance score is scaled to 0-1."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=engine,
        pre_delete_collection=True,
    )

    output = docsearch.similarity_search_with_relevance_scores("foo", k=3)
    prune_document_ids(output)
    # Original score values: 1.0, 0.9996744261675065, 0.9986996093328621
    assert output == [
        (Document(page_content="foo", metadata={"page": "0"}), 1.0),
        (Document(page_content="bar", metadata={"page": "1"}), 0.5),
        (Document(page_content="baz", metadata={"page": "2"}), 0.2),
    ]


def test_cratedb_retriever_search_threshold(engine: sa.Engine) -> None:
    """Test using retriever for searching with threshold."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=engine,
        pre_delete_collection=True,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.35},  # Original value: 0.999
    )
    output = retriever.invoke("summer")
    prune_document_ids(output)
    assert output == [
        Document(page_content="foo", metadata={"page": "0"}),
        Document(page_content="bar", metadata={"page": "1"}),
    ]


def test_cratedb_retriever_search_threshold_custom_normalization_fn(
    engine: sa.Engine,
) -> None:  # noqa: E501
    """Test searching with threshold and custom normalization function"""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        metadatas=metadatas,
        connection=engine,
        pre_delete_collection=True,
        relevance_score_fn=lambda d: d * 0,
    )

    retriever = docsearch.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.5},
    )
    output = retriever.invoke("foo")
    assert output == []


def test_cratedb_max_marginal_relevance_search(engine: sa.Engine) -> None:
    """Test max marginal relevance search."""
    texts = ["foo", "bar", "baz"]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection=engine,
        pre_delete_collection=True,
    )
    output = docsearch.max_marginal_relevance_search("foo", k=1, fetch_k=3)
    prune_document_ids(output)
    assert output == [Document(page_content="foo")]


def test_cratedb_max_marginal_relevance_search_with_score(engine: sa.Engine) -> None:
    """Test max marginal relevance search with relevance scores."""
    texts = ["foo", "bar", "baz"]
    docsearch = CrateDBVectorStore.from_texts(
        texts=texts,
        collection_name="test_collection",
        embedding=FakeEmbeddingsWithAdaDimension(),
        connection=engine,
        pre_delete_collection=True,
    )
    output = docsearch.max_marginal_relevance_search_with_score("foo", k=1, fetch_k=3)
    prune_document_ids(output)
    assert output == [(Document(page_content="foo"), 1.0)]


# We should reuse this test-case across other integrations
# Add database fixture using pytest
@pytest.fixture
def cratedb() -> Generator[CrateDBVectorStore, None, None]:
    """Create an instance of CrateDBVectorStore."""
    with get_vectorstore() as vector_store:
        yield vector_store


@contextlib.contextmanager
def get_vectorstore(
    *, embedding: Optional[Embeddings] = None
) -> Generator[CrateDBVectorStore, None, None]:
    """Get a pre-populated-vectorstore"""
    store = CrateDBVectorStore.from_documents(
        documents=DOCUMENTS,
        collection_name="test_collection",
        embedding=embedding or FakeEmbeddingsWithAdaDimension(),
        connection=CONNECTION_STRING,
        pre_delete_collection=True,
        relevance_score_fn=lambda d: d * 0,
    )
    try:
        yield cast(CrateDBVectorStore, store)
    finally:
        store.drop_tables()


@pytest.mark.parametrize("test_filter, expected_ids", TYPE_1_FILTERING_TEST_CASES)
def test_cratedb_with_metadata_filters_1(
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    """Test end to end construction and search."""
    with get_vectorstore() as cratedb:
        docs = cratedb.similarity_search("meow", k=5, filter=test_filter)
        assert [doc.metadata["id"] for doc in docs] == expected_ids, test_filter


@pytest.mark.parametrize("test_filter, expected_ids", TYPE_2_FILTERING_TEST_CASES)
def test_cratedb_with_metadata_filters_2(
    cratedb: CrateDBVectorStore,
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    """Test end to end construction and search."""
    docs = cratedb.similarity_search("meow", k=5, filter=test_filter)
    assert [doc.metadata["id"] for doc in docs] == expected_ids, test_filter


@pytest.mark.parametrize("test_filter, expected_ids", TYPE_3_FILTERING_TEST_CASES)
def test_cratedb_with_metadata_filters_3(
    cratedb: CrateDBVectorStore,
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    """Test end to end construction and search."""
    docs = cratedb.similarity_search("meow", k=5, filter=test_filter)
    assert [doc.metadata["id"] for doc in docs] == expected_ids, test_filter


@pytest.mark.parametrize("test_filter, expected_ids", TYPE_4_FILTERING_TEST_CASES)
def test_cratedb_with_metadata_filters_4(
    cratedb: CrateDBVectorStore,
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    """Test end to end construction and search."""
    docs = cratedb.similarity_search("meow", k=5, filter=test_filter)
    assert [doc.metadata["id"] for doc in docs] == expected_ids, test_filter


@pytest.mark.parametrize("test_filter, expected_ids", TYPE_5_FILTERING_TEST_CASES)
def test_cratedb_with_metadata_filters_5(
    cratedb: CrateDBVectorStore,
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    """Test end to end construction and search."""
    docs = cratedb.similarity_search("meow", k=5, filter=test_filter)
    assert [doc.metadata["id"] for doc in docs] == expected_ids, test_filter


# TODO: Missing at upstream `langchain-postgres`.
@pytest.mark.parametrize("test_filter, expected_ids", TYPE_6_FILTERING_TEST_CASES)
def test_cratedb_with_metadata_filters_6(
    cratedb: CrateDBVectorStore,
    test_filter: Dict[str, Any],
    expected_ids: List[int],
) -> None:
    """Test end to end construction and search."""
    docs = cratedb.similarity_search("meow", k=5, filter=test_filter)
    assert [doc.metadata["id"] for doc in docs] == expected_ids, test_filter


@pytest.mark.parametrize(
    "invalid_filter",
    [
        ["hello"],
        {
            "id": 2,
            "$name": "foo",
        },
        {"$or": {}},
        {"$and": {}},
        {"$between": {}},
        {"$eq": {}},
        {"$exists": {}},
        {"$exists": 1},
        {"$not": 2},
    ],
)
def test_invalid_filters(cratedb: CrateDBVectorStore, invalid_filter: Any) -> None:
    """Verify that invalid filters raise an error."""
    with pytest.raises(ValueError):
        cratedb._create_filter_clause(invalid_filter)
