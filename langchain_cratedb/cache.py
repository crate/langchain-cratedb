import typing as t

import sqlalchemy as sa
from langchain_community.cache import FullLLMCache, SQLAlchemyCache, _hash
from langchain_core.caches import RETURN_VAL_TYPE, BaseCache
from langchain_core.embeddings import Embeddings
from langchain_core.load import dumps, loads
from langchain_core.outputs import Generation
from sqlalchemy_cratedb.support import refresh_after_dml

from langchain_cratedb.vectorstores import CrateDBVectorStore
from langchain_cratedb.vectorstores.main import DBConnection


class CrateDBCache(SQLAlchemyCache):
    """
    CrateDB adapter for LangChain standard / full cache subsystem.
    It builds upon SQLAlchemyCache 1:1.
    """

    def __init__(
        self, engine: sa.Engine, cache_schema: t.Type[FullLLMCache] = FullLLMCache
    ):
        refresh_after_dml(engine)
        super().__init__(engine, cache_schema)


class CrateDBSemanticCache(BaseCache):
    """
    CrateDB adapter for LangChain semantic cache subsystem.
    It uses CrateDBVectorStore as a backend.
    """

    def __init__(
        self,
        embedding: Embeddings,
        *,
        connection: t.Union[
            None, DBConnection, sa.Engine, sa.ext.asyncio.AsyncEngine, str
        ] = None,
        cache_table_prefix: str = "cache_",
        search_threshold: float = 0.2,
        **kwargs: t.Any,
    ):
        """Initialize with necessary components.

        Args:
            embedding (Embeddings): A text embedding model.
            cache_table_prefix (str, optional): Prefix for the cache table name.
                Defaults to "cache_".
            search_threshold (float, optional): The minimum similarity score for
                a search result to be considered a match. Defaults to 0.2.

        Examples:
            Basic Usage:

            .. code-block:: python

                import langchain
                from langchain_cratedb import CrateDBSemanticCache
                from langchain.embeddings import OpenAIEmbeddings

                langchain.llm_cache = CrateDBSemanticCache(
                    embedding=OpenAIEmbeddings(),
                    host="https://user:password@127.0.0.1:4200/?schema=testdrive"
                )

            Advanced Usage:

            .. code-block:: python

                import langchain
                from langchain_cratedb import CrateDBSemanticCache
                from langchain.embeddings import OpenAIEmbeddings

                langchain.llm_cache = = CrateDBSemanticCache(
                    embeddings=OpenAIEmbeddings(),
                    host="127.0.0.1",
                    port=4200,
                    user="user",
                    password="password",
                    database="crate",
                )
        """

        self._cache_dict: t.Dict[str, CrateDBVectorStore] = {}
        self.embedding = embedding
        self.connection = connection
        self.cache_table_prefix = cache_table_prefix
        self.search_threshold = search_threshold

        # Pass the rest of the kwargs to the connection.
        self.connection_kwargs = kwargs

    def _index_name(self, llm_string: str) -> str:
        hashed_index = _hash(llm_string)
        return f"{self.cache_table_prefix}{hashed_index}"

    def _get_llm_cache(self, llm_string: str) -> CrateDBVectorStore:
        index_name = self._index_name(llm_string)

        # return vectorstore client for the specific llm string
        if index_name not in self._cache_dict:
            vs = self._cache_dict[index_name] = CrateDBVectorStore(
                embeddings=self.embedding,
                connection=self.connection,
                collection_name=index_name,
                **self.connection_kwargs,
            )
            _embedding = self.embedding.embed_query(text="test")
            vs._init_models(_embedding)
            vs.create_tables_if_not_exists()
        llm_cache = self._cache_dict[index_name]
        llm_cache.create_collection()
        return llm_cache

    def lookup(self, prompt: str, llm_string: str) -> t.Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        llm_cache = self._get_llm_cache(llm_string)
        generations: t.List = []
        # Read from a Hash
        results = llm_cache.similarity_search_with_score(
            query=prompt,
            k=1,
        )
        """
        from langchain_postgres.vectorstores import DistanceStrategy
        if llm_cache.distance_strategy != DistanceStrategy.EUCLIDEAN:
            raise NotImplementedError(f"CrateDB's vector store only implements Euclidean distance. "
                                      f"Your selection was: {llm_cache.distance_strategy}")
        """  # noqa: E501
        if results:
            for document_score in results:
                if document_score[1] <= self.search_threshold:
                    generations.extend(loads(document_score[0].metadata["return_val"]))
        return generations if generations else None

    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(
                    "CrateDBSemanticCache only supports caching of "
                    f"normal LLM generations, got {type(gen)}"
                )
        llm_cache = self._get_llm_cache(llm_string)
        metadata = {
            "llm_string": llm_string,
            "prompt": prompt,
            "return_val": dumps(list(return_val)),
        }
        llm_cache.add_texts(texts=[prompt], metadatas=[metadata])

    def clear(self, **kwargs: t.Any) -> None:
        """Clear semantic cache for a given llm_string."""
        if "llm_string" in kwargs:
            index_name = self._index_name(kwargs["llm_string"])
            if index_name in self._cache_dict:
                vs = self._cache_dict[index_name]
                with vs._make_sync_session() as session:
                    collection = vs.get_collection(session)
                    collection.embeddings.clear()
                    session.commit()
                del self._cache_dict[index_name]
        else:
            raise NotImplementedError(
                "Clearing cache elements without constraints is not implemented yet"
            )
