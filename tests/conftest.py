import typing as t

import pytest
import sqlalchemy as sa

from langchain_cratedb.vectorstores.model import ModelFactory
from tests.settings import CONNECTION_STRING
from tests.util.python import run_file  # noqa: F401


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
