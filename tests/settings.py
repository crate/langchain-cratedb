import os

from langchain_cratedb import CrateDBVectorStore

SCHEMA_NAME = os.environ.get("TEST_CRATEDB_DATABASE", "testdrive")

CONNECTION_STRING = CrateDBVectorStore.connection_string_from_db_params(
    driver=os.environ.get("TEST_CRATEDB_DRIVER", "crate"),
    host=os.environ.get("TEST_CRATEDB_HOST", "localhost"),
    port=int(os.environ.get("TEST_CRATEDB_PORT", "4200")),
    database=SCHEMA_NAME,
    user=os.environ.get("TEST_CRATEDB_USER", "crate"),
    password=os.environ.get("TEST_CRATEDB_PASSWORD", ""),
)
