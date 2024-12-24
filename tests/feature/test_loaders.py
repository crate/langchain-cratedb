"""
Test SQLAlchemy document loader functionality on behalf of CrateDB.
"""

import functools
import logging

import pytest
import sqlalchemy as sa
from langchain_community.document_loaders.sql_database import SQLDatabaseLoader
from langchain_community.utilities.sql_database import SQLDatabase

from langchain_cratedb import CrateDBLoader
from tests.data import MLB_TEAMS_2012_SQL

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture()
def db(engine: sa.Engine) -> SQLDatabase:
    return SQLDatabase(engine=engine)


@pytest.fixture()
def provision_database(engine: sa.Engine) -> None:
    """
    Provision database with table schema and data.
    """
    sql_statements = MLB_TEAMS_2012_SQL.read_text()
    with engine.connect() as connection:
        connection.execute(sa.text("DROP TABLE IF EXISTS mlb_teams_2012;"))
        for statement in sql_statements.split(";"):
            statement = statement.strip()
            if not statement:
                continue
            connection.execute(sa.text(statement))
            connection.commit()
        if engine.dialect.name.startswith("crate"):
            connection.execute(sa.text("REFRESH TABLE mlb_teams_2012;"))
            connection.commit()


def test_cratedb_loader_no_options(db: SQLDatabase) -> None:
    """Test SQLAlchemy loader basics."""

    loader = CrateDBLoader("SELECT 1 AS a, 2 AS b", db=db)
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "a: 1\nb: 2"
    assert docs[0].metadata == {}


def test_cratedb_loader_include_rownum_into_metadata(db: SQLDatabase) -> None:
    """Test SQLAlchemy loader with row number in metadata."""

    loader = CrateDBLoader(
        "SELECT 1 AS a, 2 AS b",
        db=db,
        include_rownum_into_metadata=True,
    )
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "a: 1\nb: 2"
    assert docs[0].metadata == {"row": 0}


def test_cratedb_loader_include_query_into_metadata(db: SQLDatabase) -> None:
    """Test SQLAlchemy loader with query in metadata."""

    loader = CrateDBLoader(
        "SELECT 1 AS a, 2 AS b", db=db, include_query_into_metadata=True
    )
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].page_content == "a: 1\nb: 2"
    assert docs[0].metadata == {"query": "SELECT 1 AS a, 2 AS b"}


def test_cratedb_loader_page_content_columns(db: SQLDatabase) -> None:
    """Test SQLAlchemy loader with defined page content columns."""

    # Define a custom callback function to convert a row into a "page content" string.
    row_to_content = functools.partial(
        SQLDatabaseLoader.page_content_default_mapper, column_names=["a"]
    )

    loader = CrateDBLoader(
        "SELECT 1 AS a, 2 AS b UNION SELECT 3 AS a, 4 AS b",
        db=db,
        page_content_mapper=row_to_content,
    )
    docs = loader.load()

    assert len(docs) == 2
    assert docs[0].page_content == "a: 1"
    assert docs[0].metadata == {}

    assert docs[1].page_content == "a: 3"
    assert docs[1].metadata == {}


def test_cratedb_loader_metadata_columns(db: SQLDatabase) -> None:
    """Test SQLAlchemy loader with defined metadata columns."""

    # Define a custom callback function to convert a row into a "metadata" dictionary.
    row_to_metadata = functools.partial(
        SQLDatabaseLoader.metadata_default_mapper, column_names=["b"]
    )

    loader = CrateDBLoader(
        "SELECT 1 AS a, 2 AS b",
        db=db,
        metadata_mapper=row_to_metadata,
    )
    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].metadata == {"b": 2}


def test_cratedb_loader_real_data_with_sql_no_parameters(
    db: SQLDatabase, provision_database: None
) -> None:
    """Test SQLAlchemy loader with real data, querying by SQL statement."""

    loader = CrateDBLoader(
        query='SELECT * FROM mlb_teams_2012 ORDER BY "Team";',
        db=db,
    )
    docs = loader.load()

    assert len(docs) == 30
    assert docs[0].page_content == "Team: Angels\nPayroll (millions): 154.49\nWins: 89"
    assert docs[0].metadata == {}


def test_cratedb_loader_real_data_with_sql_and_parameters(
    db: SQLDatabase, provision_database: None
) -> None:
    """Test SQLAlchemy loader, querying by SQL statement and parameters."""

    loader = CrateDBLoader(
        query='SELECT * FROM mlb_teams_2012 WHERE "Team" LIKE :search ORDER BY "Team";',
        parameters={"search": "R%"},
        db=db,
    )
    docs = loader.load()

    assert len(docs) == 6
    assert docs[0].page_content == "Team: Rangers\nPayroll (millions): 120.51\nWins: 93"
    assert docs[0].metadata == {}


def test_cratedb_loader_real_data_with_selectable(
    db: SQLDatabase, provision_database: None
) -> None:
    """Test SQLAlchemy loader with real data, querying by SQLAlchemy selectable."""

    # Define an SQLAlchemy table.
    mlb_teams_2012 = sa.Table(
        "mlb_teams_2012",
        sa.MetaData(),
        sa.Column("Team", sa.VARCHAR),
        sa.Column("Payroll (millions)", sa.FLOAT),
        sa.Column("Wins", sa.BIGINT),
    )

    # Query the database table using an SQLAlchemy selectable.
    select = sa.select(mlb_teams_2012).order_by(mlb_teams_2012.c.Team)
    loader = CrateDBLoader(
        query=select,
        db=db,
        include_query_into_metadata=True,
    )
    docs = loader.load()

    assert len(docs) == 30
    assert docs[0].page_content == "Team: Angels\nPayroll (millions): 154.49\nWins: 89"
    assert docs[0].metadata == {
        "query": 'SELECT mlb_teams_2012."Team", mlb_teams_2012."Payroll (millions)", '
        'mlb_teams_2012."Wins" \nFROM mlb_teams_2012 '
        'ORDER BY mlb_teams_2012."Team"'
    }
