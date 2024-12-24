"""CrateDB vector stores."""

from __future__ import annotations

import contextlib
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from typing import (
    cast as typing_cast,
)

import sqlalchemy as sa
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_postgres._utils import maximal_marginal_relevance
from langchain_postgres.vectorstores import (
    _LANGCHAIN_DEFAULT_COLLECTION_NAME,  # noqa: F401
    LOGICAL_OPERATORS,
    SPECIAL_CASED_OPERATORS,
    TEXT_OPERATORS,
    DistanceStrategy,
    PGVector,
)
from sqlalchemy_cratedb.support import refresh_table

from langchain_cratedb.vectorstores.model import ModelFactory

# CrateDB and Lucene currently only implement
# similarity based on the Euclidean distance.
#
# > Today, when creating a FLOAT_VECTOR, it uses the default
# > EUCLIDEAN_HNSW (L2) similarity.
# >
# > -- https://github.com/crate/crate/issues/15768
DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.EUCLIDEAN

COMPARISONS_TO_NATIVE = {
    "$eq": "=",
    "$ne": "!=",
    "$lt": "<",
    "$lte": "<=",
    "$gt": ">",
    "$gte": ">=",
}

SUPPORTED_OPERATORS = (
    set(COMPARISONS_TO_NATIVE)
    .union(TEXT_OPERATORS)
    .union(LOGICAL_OPERATORS)
    .union(SPECIAL_CASED_OPERATORS)
)


VST = TypeVar("VST", bound=VectorStore)
DBConnection = Union[sa.engine.Engine, str]


class CrateDBVectorStore(PGVector):
    # TODO: Replace all TODOs in docstring.
    """CrateDB vector store integration.

    # TODO: Replace with relevant packages, env vars.
    Setup:
        Install ``langchain-cratedb`` and set environment variable ``CRATEDB_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-cratedb
            export CRATEDB_API_KEY="your-api-key"

    # TODO: Populate with relevant params.
    Key init args — indexing params:
        collection_name: str
            Name of the collection.
        embedding_function: Embeddings
            Embedding function to use.

    # TODO: Populate with relevant params.
    Key init args — client params:
        client: Optional[Client]
            Client to use.
        connection_args: Optional[dict]
            Connection arguments.

    # TODO: Replace with relevant init params.
    Instantiate:
        .. code-block:: python

            from langchain_cratedb.vectorstores import CrateDBVectorStore
            from langchain_openai import OpenAIEmbeddings

            vector_store = CrateDBVectorStore(
                collection_name="foo",
                embedding_function=OpenAIEmbeddings(),
                connection_args={"uri": "./foo.db"},
                # other params...
            )

    # TODO: Populate with relevant variables.
    Add Documents:
        .. code-block:: python

            from langchain_core.documents import Document

            document_1 = Document(page_content="foo", metadata={"baz": "bar"})
            document_2 = Document(page_content="thud", metadata={"bar": "baz"})
            document_3 = Document(page_content="i will be deleted :(")

            documents = [document_1, document_2, document_3]
            ids = ["1", "2", "3"]
            vector_store.add_documents(documents=documents, ids=ids)

    # TODO: Populate with relevant variables.
    Delete Documents:
        .. code-block:: python

            vector_store.delete(ids=["3"])

    # TODO: Fill out with relevant variables and example output.
    Search:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud",k=1)
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            # TODO: Example output

    # TODO: Fill out with relevant variables and example output.
    Search with filter:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud",k=1,filter={"bar": "baz"})
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            # TODO: Example output

    # TODO: Fill out with relevant variables and example output.
    Search with score:
        .. code-block:: python

            results = vector_store.similarity_search_with_score(query="qux",k=1)
            for doc, score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            # TODO: Example output

    # TODO: Fill out with relevant variables and example output.
    Async:
        .. code-block:: python

            # add documents
            # await vector_store.aadd_documents(documents=documents, ids=ids)

            # delete documents
            # await vector_store.adelete(ids=["3"])

            # search
            # results = vector_store.asimilarity_search(query="thud",k=1)

            # search with score
            results = await vector_store.asimilarity_search_with_score(query="qux",k=1)
            for doc,score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            # TODO: Example output

    # TODO: Fill out with relevant variables and example output.
    Use as Retriever:
        .. code-block:: python

            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
            )
            retriever.invoke("thud")

        .. code-block:: python

            # TODO: Example output

    """  # noqa: E501

    @classmethod
    def connection_string_from_db_params(
        cls,
        driver: str,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
    ) -> str:
        """Return connection string from database parameters."""
        if driver != "crate":
            raise NotImplementedError("Only sqlalchemy-cratedb driver is supported")
        return f"{driver}://{user}:{password}@{host}:{port}/?schema={database}"

    def __post_init__(
        self,
    ) -> None:
        """
        Initialize the store.
        """

        # Need to defer initialization, because dimension size
        # can only be figured out at runtime.
        self.BaseModel = None
        self.CollectionStore = None
        self.EmbeddingStore = None

    @contextlib.contextmanager
    def _make_sync_session(self) -> Generator[sa.orm.Session, None, None]:
        """Make an async session."""
        if self.async_mode:
            raise ValueError(
                "Attempting to use a sync method in when async mode is turned on. "
                "Please use the corresponding async method instead."
            )
        with self.session_maker() as session:
            # Patch dialect to invoke `REFRESH TABLE` after each DML operation.
            from sqlalchemy_cratedb.support import refresh_after_dml

            refresh_after_dml(session.get_bind())

            yield typing_cast(sa.orm.Session, session)

    def _init_models(self, embedding: List[float]) -> None:
        """
        With CrateDB, vector dimensionality is obligatory, so create tables at runtime.

        Tables need to be created at runtime, because the `EmbeddingStore.embedding`
        field, a `FloatVector`, MUST be initialized with a dimensionality
        parameter, which is only obtained at runtime. This is different when
        compared with other vector stores.
        """

        # TODO: Use a better way to run this only once.
        if self.CollectionStore is not None and self.EmbeddingStore is not None:
            return

        size = len(embedding)
        self._init_models_with_dimensionality(size=size)

    def _init_models_with_dimensionality(self, size: int) -> None:
        mf = ModelFactory(dimensions=size)
        self.BaseModel, self.CollectionStore, self.EmbeddingStore = (
            mf.BaseModel,  # type: ignore[assignment]
            mf.CollectionStore,
            mf.EmbeddingStore,
        )

    def create_tables_if_not_exists(self) -> None:
        """
        Need to overwrite because this `Base` is different from parent's `Base`.
        """
        if self.BaseModel is None:
            raise RuntimeError("Storage models not initialized")
        with self._make_sync_session() as session:
            self.BaseModel.metadata.create_all(session.get_bind())
            session.commit()

    def delete(
        self,
        ids: Optional[List[str]] = None,
        collection_only: bool = False,
        **kwargs: Any,
    ) -> None:
        """Delete vectors by ids or uuids.

        Args:
            ids: List of ids to delete.
            collection_only: Only delete ids in the collection.
        """

        # CrateDB: Calling ``delete`` must not raise an exception
        #          when deleting IDs that do not exist.
        if self.EmbeddingStore is None:
            return None
        return super().delete(ids=ids, collection_only=collection_only, **kwargs)

    def _ensure_storage(self) -> None:
        """
        With CrateDB, vector dimensionality is obligatory, so create tables at runtime.

        Tables need to be created at runtime, because the `EmbeddingStore.embedding`
        field, a `FloatVector`, needs to be initialized with a dimensionality
        parameter, which is only obtained at runtime.
        """
        self.create_tables_if_not_exists()
        self.create_collection()

    def get_collection(self, session: sa.orm.Session) -> Any:
        if self.CollectionStore is None:
            raise RuntimeError(
                "Collection can't be accessed without specifying "
                "dimension size of embedding vectors"
            )
        try:
            return self.CollectionStore.get_by_name(session, self.collection_name)
        # TODO: Q&A: Must not raise an exception when collection does not exist?
        except sa.exc.ProgrammingError as ex:
            if "RelationUnknown" not in str(ex):
                raise

    def add_embeddings(
        self,
        texts: Sequence[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add embeddings to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            embeddings: List of list of embedding vectors.
            metadatas: List of metadatas associated with the texts.
            kwargs: vectorstore specific parameters
        """

        if not embeddings:
            return []
        self._init_models(embeddings[0])

        # When the user requested to delete the collection before running subsequent
        # operations on it, run the deletion gracefully if the table does not exist
        # yet.
        if self.pre_delete_collection:
            try:
                self.delete_collection()
            # TODO: Q&A: Must not raise an exception when collection does not exist?
            except sa.exc.ProgrammingError as ex:
                if "RelationUnknown" not in str(ex):
                    raise

        # CrateDB: Tables need to be created at runtime.
        self._ensure_storage()

        # After setting up the table/collection at runtime, add embeddings.
        embedding_ids = super().add_embeddings(
            texts=texts, embeddings=embeddings, metadatas=metadatas, ids=ids, **kwargs
        )
        with self._make_sync_session() as session:
            refresh_table(session, self.EmbeddingStore)
        return embedding_ids

    def _results_to_docs_and_scores(self, results: Any) -> List[Tuple[Document, float]]:
        """Return docs and scores from results."""
        return [
            (
                Document(
                    id=str(result.EmbeddingStore.id),
                    page_content=result.EmbeddingStore.document,
                    metadata=result.EmbeddingStore.cmetadata,
                ),
                result.similarity if self.embedding_function is not None else None,
            )
            for result in results
        ]

    def get_by_ids(self, ids: Sequence[str], /) -> List[Document]:
        """Get documents by ids."""
        # ``get_by_ids`` must be implemented and must not
        # raise an exception when given IDs that do not exist.
        if self.EmbeddingStore is None:
            return []
        return super().get_by_ids(ids)

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """
        The 'correct' relevance function
        may differ depending on a few things, including:
        - the distance / similarity metric used by the VectorStore
        - the scale of your embeddings (OpenAI's are unit normed. Many others are not!)
        - embedding dimensionality
        - etc.
        """
        if self.override_relevance_score_fn is not None:
            return self.override_relevance_score_fn

        # TODO: Always select EUCLIDEAN, because CrateDB does not provide support for
        #       the others. Unfortunately, langchain-postgres's default
        #       `DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE` can't easily be
        #       changed.
        return self._euclidean_relevance_score_fn

    @staticmethod
    def _euclidean_relevance_score_fn(similarity: float) -> float:
        """Return a similarity score on a scale [0, 1]."""
        # The 'correct' relevance function
        # may differ depending on a few things, including:
        # - the distance / similarity metric used by the VectorStore
        # - the scale of your embeddings (OpenAI's are unit normed. Many
        #  others are not!)
        # - embedding dimensionality
        # - etc.
        # This function converts the Euclidean norm of normalized embeddings
        # (0 is most similar, sqrt(2) most dissimilar)
        # to a similarity function (0 to 1)

        # CrateDB uses the `vector_similarity()` SQL function in this context,
        # which already returns a normalized value.
        # https://cratedb.com/docs/crate/reference/en/latest/general/builtins/scalar-functions.html#vector-similarity-float-vector-float-vector
        return similarity

    @staticmethod
    def _cosine_relevance_score_fn(distance: float) -> float:
        """
        Normalize the distance to a score on a scale [0, 1], using Cosine similarity.
        Not supported by CrateDB.
        """
        raise NotImplementedError("CrateDB does not support Cosine similarity")

    @staticmethod
    def _max_inner_product_relevance_score_fn(distance: float) -> float:
        """
        Normalize the distance to a score on a scale [0, 1], using Dot-product.
        Not supported by CrateDB.
        """
        raise NotImplementedError("CrateDB does not support Dot-product similarity")

    ### NEED TO OVERWRITE BECAUSE __query_collection ###

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict] = None,  # noqa: A002
    ) -> List[Tuple[Document, float]]:
        assert not self._async_engine, "This method must be called without async_mode"  # noqa: S101
        results = self.__query_collection(embedding=embedding, k=k, filter=filter)

        return self._results_to_docs_and_scores(results)

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, str]] = None,  # noqa: A002
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs selected using the maximal marginal relevance with score
            to embedding vector.

        Maximal marginal relevance optimizes for similarity to query AND diversity
            among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k (int): Number of Documents to return. Defaults to 4.
            fetch_k (int): Number of Documents to fetch to pass to MMR algorithm.
                Defaults to 20.
            lambda_mult (float): Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding
                to maximum diversity and 1 to minimum diversity.
                Defaults to 0.5.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.

        Returns:
            List[Tuple[Document, float]]: List of Documents selected by maximal marginal
                relevance to the query and score for each.
        """
        import numpy as np

        assert not self._async_engine, "This method must be called without async_mode"  # noqa: S101
        results = self.__query_collection(embedding=embedding, k=fetch_k, filter=filter)

        embedding_list = [result.EmbeddingStore.embedding for result in results]

        mmr_selected = maximal_marginal_relevance(
            np.array(embedding, dtype=np.float32),
            embedding_list,
            k=k,
            lambda_mult=lambda_mult,
        )

        candidates = self._results_to_docs_and_scores(results)

        return [r for i, r in enumerate(candidates) if i in mmr_selected]

    def __query_collection(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,  # noqa: A002
    ) -> List[Any]:
        """Query the collection."""
        self._init_models(embedding)
        with self._make_sync_session() as session:
            collection = self.get_collection(session)
            if collection is None:
                raise ValueError(f"Collection not found: {self.collection_name}")
            return self._query_collection_multi(
                collections=[collection], embedding=embedding, k=k, filter=filter
            )

    def _query_collection_multi(
        self,
        collections: List[Any],
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, str]] = None,  # noqa: A002
    ) -> List[Any]:
        """Query the collection."""
        self._init_models(embedding)

        collection_names = [coll.name for coll in collections]
        collection_uuids = [coll.uuid for coll in collections]
        self.logger.info(f"Querying collections: {collection_names}")

        with self._make_sync_session() as session:
            filter_by = [self.EmbeddingStore.collection_id.in_(collection_uuids)]

            if filter is not None:
                filter_clause = self._create_filter_clause(filter)
                if filter_clause is not None:
                    filter_by.append(filter_clause)

            _type = self.EmbeddingStore

            results: List[Any] = (
                session.query(
                    self.EmbeddingStore,
                    # TODO: Original pgvector code uses `self.distance_strategy`.
                    #       CrateDB currently only supports EUCLIDEAN.
                    #       self.distance_strategy(embedding).label("distance")  # noqa: E501,ERA001
                    sa.func.vector_similarity(
                        self.EmbeddingStore.embedding,
                        # TODO: Just reference the `embedding` symbol here, don't
                        #       serialize its value prematurely.
                        #       https://github.com/crate/crate/issues/16912
                        #
                        # Until that got fixed, marshal the arguments to
                        # `vector_similarity()` manually, in order to work around
                        # this edge case bug. We don't need to use JSON marshalling,
                        # because Python's string representation of a list is just
                        # right.
                        sa.text(str(embedding)),
                    ).label("similarity"),
                )
                .filter(*filter_by)
                # CrateDB applies `KNN_MATCH` within the `WHERE` clause.
                .filter(sa.func.knn_match(self.EmbeddingStore.embedding, embedding, k))
                .order_by(sa.desc("similarity"))
                .join(
                    self.CollectionStore,
                    self.EmbeddingStore.collection_id == self.CollectionStore.uuid,
                )
                .limit(k)
            )
        return results

    def _handle_field_filter(
        self,
        field: str,
        value: Any,
    ) -> sa.SQLColumnExpression:
        """Create a filter for a specific field.

        Needs to be overwritten for CrateDB, in order to:

        - Convert the use of `jsonb_path_match` functions into direct accessors
          to CrateDB's OBJECT type, paired with vanilla SQL expressions, in order
          to apply filter expressions to `cmetadata` columns.
          This applies to all the standard comparison operators, as well as
          `$between` and `$exists`.

        - Remove `.astext` field accessor, because this is likely a psycopg thing.
          See https://github.com/crate/sqlalchemy-cratedb/issues/188.

        Args:
            field: name of field
            value: value to filter
                If provided as is then this will be an equality filter
                If provided as a dictionary then this will be a filter, the key
                will be the operator and the value will be the value to filter by

        Returns:
            sqlalchemy expression
        """
        if not isinstance(field, str):
            raise ValueError(
                f"field should be a string but got: {type(field)} with value: {field}"
            )

        if field.startswith("$"):
            raise ValueError(
                f"Invalid filter condition. Expected a field but got an operator: "
                f"{field}"
            )

        # Allow [a-zA-Z0-9_], disallow $ for now until we support escape characters
        if not field.isidentifier():
            raise ValueError(
                f"Invalid field name: {field}. Expected a valid identifier."
            )

        if isinstance(value, dict):
            # This is a filter specification
            if len(value) != 1:
                raise ValueError(
                    "Invalid filter condition. Expected a value which "
                    "is a dictionary with a single key that corresponds to an operator "
                    f"but got a dictionary with {len(value)} keys. The first few "
                    f"keys are: {list(value.keys())[:3]}"
                )
            operator, filter_value = list(value.items())[0]
            # Translate operator syntax.  # TODO: Translate all parameters?
            if operator == "IN":
                operator = "$in"
            # Verify that that operator is an operator
            if operator not in SUPPORTED_OPERATORS:
                raise ValueError(
                    f"Invalid operator: {operator}. "
                    f"Expected one of {SUPPORTED_OPERATORS}"
                )
        else:  # Then we assume an equality operator
            operator = "$eq"
            filter_value = value

        import sqlalchemy as sa

        if operator in COMPARISONS_TO_NATIVE:
            # Then we implement an equality filter
            # native is trusted input
            native = COMPARISONS_TO_NATIVE[operator]
            return self.EmbeddingStore.cmetadata[field].op(native)(filter_value)
        if operator == "$between":
            # Use AND with two comparisons
            low, high = filter_value
            lower_bound = self.EmbeddingStore.cmetadata[field].op(">=")(low)
            upper_bound = self.EmbeddingStore.cmetadata[field].op("<=")(high)
            return sa.and_(lower_bound, upper_bound)
        if operator in {"$in", "$nin", "$like", "$ilike"}:
            # We'll do force coercion to text
            if operator in {"$in", "$nin"}:
                for val in filter_value:
                    if not isinstance(val, (str, int, float)):
                        raise NotImplementedError(
                            f"Unsupported type: {type(val)} for value: {val}"
                        )

                    if isinstance(val, bool):  # b/c bool is an instance of int
                        raise NotImplementedError(
                            f"Unsupported type: {type(val)} for value: {val}"
                        )

            queried_field = self.EmbeddingStore.cmetadata[field]

            if operator in {"$in"}:
                return queried_field.in_([str(val) for val in filter_value])
            if operator in {"$nin"}:
                return ~queried_field.in_([str(val) for val in filter_value])
            if operator in {"$like"}:
                return queried_field.like(filter_value)
            if operator in {"$ilike"}:
                return queried_field.ilike(filter_value)
            raise NotImplementedError()
        if operator == "$exists":
            if not isinstance(filter_value, bool):
                raise ValueError(
                    "Expected a boolean value for $exists "
                    f"operator, but got: {filter_value}"
                )
            condition = sa.literal(field).op("=")(
                sa.func.any(sa.func.object_keys(self.EmbeddingStore.cmetadata))
            )
            return condition if filter_value else ~condition
        raise NotImplementedError()
