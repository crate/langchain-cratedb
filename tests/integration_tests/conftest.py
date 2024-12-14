import os
import typing as t

import pytest
import sqlalchemy as sa

from langchain_cratedb import CrateDBVectorStore
from langchain_cratedb.vectorstores.model import ModelFactory

SCHEMA_NAME = os.environ.get("TEST_CRATEDB_DATABASE", "testdrive")

CONNECTION_STRING = CrateDBVectorStore.connection_string_from_db_params(
    driver=os.environ.get("TEST_CRATEDB_DRIVER", "crate"),
    host=os.environ.get("TEST_CRATEDB_HOST", "localhost"),
    port=int(os.environ.get("TEST_CRATEDB_PORT", "4200")),
    database=SCHEMA_NAME,
    user=os.environ.get("TEST_CRATEDB_USER", "crate"),
    password=os.environ.get("TEST_CRATEDB_PASSWORD", ""),
)


@pytest.fixture
def engine() -> sa.Engine:
    """
    Return an SQLAlchemy engine object.
    """
    return sa.create_engine(CONNECTION_STRING, echo=False)


@pytest.fixture
def session(engine: sa.Engine) -> t.Generator[sa.orm.Session, None, None]:
    with engine.connect() as conn:
        with sa.orm.Session(conn) as session:
            yield session


@pytest.fixture(autouse=True)
def drop_tables(engine: sa.Engine) -> None:
    """
    Drop database tables.
    """
    try:
        mf = ModelFactory()
        mf.BaseModel.metadata.drop_all(engine, checkfirst=False)
    except Exception as ex:
        if "RelationUnknown" not in str(ex):
            raise


@pytest.fixture
def prune_tables(engine: sa.Engine) -> None:
    """
    Delete data from database tables.
    """
    with engine.connect() as conn:
        with sa.orm.Session(conn) as session:
            mf = ModelFactory()
            try:
                session.query(mf.CollectionStore).delete()
            except sa.exc.ProgrammingError:
                pass
            try:
                session.query(mf.EmbeddingStore).delete()
            except sa.exc.ProgrammingError:
                pass
