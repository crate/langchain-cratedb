"""
Exercise the LangChain/CrateDB document loader.

How to use the SQL document loader, based on SQLAlchemy.

The example uses the canonical `mlb_teams_2012.csv`,
converted to SQL, see `mlb_teams_2012.sql`.

Synopsis::

    # Install prerequisites.
    pip install --upgrade langchain-cratedb langchain-community

    # Start database.
    docker run --rm -it --publish=4200:4200 crate/crate:nightly

    # Optionally set environment variable to configure CrateDB connection URL.
    export CRATEDB_SQLALCHEMY_URL="crate://crate@localhost/?schema=doc"

    # Run program.
    python examples/basic/document_loader.py
"""  # noqa: E501
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "langchain-cratedb",
# ]
# ///

import os
from pprint import pprint

import requests
import sqlparse
from langchain_community.utilities import SQLDatabase

from langchain_cratedb import CrateDBLoader

CRATEDB_SQLALCHEMY_URL = os.environ.get(
    "CRATEDB_SQLALCHEMY_URL", "crate://crate@localhost/?schema=testdrive"
)


def import_mlb_teams_2012() -> None:
    """
    Import data into database table `mlb_teams_2012`.

    TODO: Refactor into general purpose package.
    """
    db = SQLDatabase.from_uri(CRATEDB_SQLALCHEMY_URL)
    # TODO: Use new URL @ langchain-cratedb.
    url = "https://github.com/crate-workbench/langchain/raw/cratedb/docs/docs/integrations/document_loaders/example_data/mlb_teams_2012.sql"
    sql = requests.get(url).text
    for statement in sqlparse.split(sql):
        db.run(statement)
    db.run("REFRESH TABLE mlb_teams_2012")


def main() -> None:
    # Load data.
    import_mlb_teams_2012()

    db = SQLDatabase.from_uri(CRATEDB_SQLALCHEMY_URL)

    # Query data.
    loader = CrateDBLoader(
        query="SELECT * FROM mlb_teams_2012 LIMIT 3;",
        db=db,
        include_rownum_into_metadata=True,
    )
    docs = loader.load()
    pprint(docs)


if __name__ == "__main__":
    main()
