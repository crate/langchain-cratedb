import typing as t

import sqlalchemy as sa
from langchain_community.cache import FullLLMCache, SQLAlchemyCache
from sqlalchemy_cratedb.support import refresh_after_dml


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
