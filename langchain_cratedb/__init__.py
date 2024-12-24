# ruff: noqa: E402  # Module level import not at top of file
from importlib import metadata

from langchain_cratedb.patches import patch_sqlalchemy_dialect

patch_sqlalchemy_dialect()

from langchain_cratedb.cache import CrateDBCache, CrateDBSemanticCache
from langchain_cratedb.chat_history import CrateDBChatMessageHistory
from langchain_cratedb.loaders import CrateDBLoader
from langchain_cratedb.vectorstores import (
    CrateDBVectorStore,
    CrateDBVectorStoreMultiCollection,
)

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:  # pragma: no cover
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "CrateDBCache",
    "CrateDBChatMessageHistory",
    "CrateDBLoader",
    "CrateDBSemanticCache",
    "CrateDBVectorStore",
    "CrateDBVectorStoreMultiCollection",
    "__version__",
]
