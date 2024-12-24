import logging
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import sqlalchemy as sa
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres._utils import maximal_marginal_relevance

from langchain_cratedb.vectorstores.main import (
    _LANGCHAIN_DEFAULT_COLLECTION_NAME,
    DEFAULT_DISTANCE_STRATEGY,
    CrateDBVectorStore,
    DBConnection,
    DistanceStrategy,
)


class CrateDBVectorStoreMultiCollection(CrateDBVectorStore):
    """
    Provide functionality for searching multiple collections.
    It can not be used for indexing documents.

    To use it, you should have the ``sqlalchemy-cratedb`` Python package installed.

    Synopsis::

        from langchain_community.vectorstores.cratedb import CrateDBVectorStoreMultiCollection

        multisearch = CrateDBVectorStoreMultiCollection(
            collection_names=["collection_foo", "collection_bar"],
            embedding_function=embeddings,
            connection_string=CONNECTION_STRING,
        )
        docs_with_score = multisearch.similarity_search_with_score(query)
    """  # noqa: E501

    def __init__(
        self,
        embeddings: Embeddings,
        *,
        connection: Union[
            None, DBConnection, sa.Engine, sa.ext.asyncio.AsyncEngine, str
        ] = None,
        embedding_length: Optional[int] = None,
        collection_names: Optional[List[str]] = None,
        collection_metadata: Optional[dict] = None,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        pre_delete_collection: bool = False,
        logger: Optional[logging.Logger] = None,
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        engine_args: Optional[dict[str, Any]] = None,
        use_jsonb: bool = True,
        create_extension: bool = True,
        async_mode: bool = False,
    ) -> None:
        """Initialize the PGVector store.
        For an async version, use `PGVector.acreate()` instead.

        Args:
            connection: Postgres connection string or (async)engine.
            embeddings: Any embedding function implementing
                `langchain.embeddings.base.Embeddings` interface.
            embedding_length: The length of the embedding vector. (default: None)
                NOTE: This is not mandatory. Defining it will prevent vectors of
                any other size to be added to the embeddings table but, without it,
                the embeddings can't be indexed.
            collection_name: The name of the collection to use. (default: langchain)
                NOTE: This is not the name of the table, but the name of the collection.
                The tables will be created when initializing the store (if not exists)
                So, make sure the user has the right permissions to create tables.
            distance_strategy: The distance strategy to use. (default: COSINE)
            pre_delete_collection: If True, will delete the collection if it exists.
                (default: False). Useful for testing.
            engine_args: SQLAlchemy's create engine arguments.
            use_jsonb: Only provided for compatibility with the PostgreSQL adapter.
                On CrateDB, metadata is stored as OBJECT.
            create_extension: If True, will create the vector extension if it
                doesn't exist. disabling creation is useful when using ReadOnly
                Databases.
        """
        self.async_mode = async_mode
        self.embedding_function = embeddings
        self._embedding_length = embedding_length
        self.collection_names = collection_names or [_LANGCHAIN_DEFAULT_COLLECTION_NAME]
        self.collection_metadata = collection_metadata
        self._distance_strategy = distance_strategy
        self.pre_delete_collection = pre_delete_collection
        self.logger = logger or logging.getLogger(__name__)
        self.override_relevance_score_fn = relevance_score_fn
        self._engine: Optional[sa.Engine] = None
        self._async_engine: Optional[sa.ext.asyncio.AsyncEngine] = None
        self._async_init = False

        if isinstance(connection, str):
            if async_mode:
                self._async_engine = sa.ext.asyncio.create_async_engine(
                    connection, **(engine_args or {})
                )
            else:
                self._engine = sa.create_engine(url=connection, **(engine_args or {}))
        elif isinstance(connection, sa.Engine):
            self.async_mode = False
            self._engine = connection
        elif isinstance(connection, sa.ext.asyncio.AsyncEngine):
            self.async_mode = True
            self._async_engine = connection
        else:
            raise ValueError(
                "connection should be a connection string or an instance of "
                "sqlalchemy.engine.Engine or sqlalchemy.ext.asyncio.engine.AsyncEngine"
            )
        self.session_maker: Union[
            sa.orm.scoped_session, sa.ext.asyncio.async_sessionmaker
        ]
        if self.async_mode:
            self.session_maker = sa.ext.asyncio.async_sessionmaker(
                bind=self._async_engine
            )
        else:
            self.session_maker = sa.orm.scoped_session(
                sa.orm.sessionmaker(bind=self._engine)
            )

        self.use_jsonb = use_jsonb
        self.create_extension = create_extension

        if not self.async_mode:
            self.__post_init__()

    def get_collections(self, session: sa.orm.Session) -> Any:
        if self.CollectionStore is None:
            raise RuntimeError(
                "Collection can't be accessed without specifying "
                "dimension size of embedding vectors"
            )
        return self.CollectionStore.get_by_names(session, self.collection_names)

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
        """Query multiple collections."""
        self._init_models(embedding)
        with self._make_sync_session() as session:
            collections = self.get_collections(session)
            if not collections:
                raise ValueError("No collections found")
            return self._query_collection_multi(
                collections=collections, embedding=embedding, k=k, filter=filter
            )

    @classmethod
    def from_texts(
        cls: Type["CrateDBVectorStoreMultiCollection"],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        collection_name: str = _LANGCHAIN_DEFAULT_COLLECTION_NAME,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        ids: Optional[List[str]] = None,
        pre_delete_collection: bool = False,
        use_jsonb: bool = True,
        **kwargs: Any,
    ) -> "CrateDBVectorStoreMultiCollection":
        """Return VectorStore initialized from documents and embeddings."""
        raise NotImplementedError(
            "The adapter for querying multiple collections "
            "can not be used for _indexing_ documents"
        )

    @classmethod
    def __from(cls, *args: List, **kwargs: Dict):  # type: ignore[no-untyped-def,override]
        raise NotImplementedError(
            "The adapter for querying multiple collections "
            "can not be used for _indexing_ documents"
        )
