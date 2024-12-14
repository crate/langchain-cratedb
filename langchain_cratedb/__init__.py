from importlib import metadata

from langchain_cratedb.document_loaders import CrateDBLoader
from langchain_cratedb.retrievers import CrateDBRetriever
from langchain_cratedb.vectorstores import CrateDBVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "CrateDBVectorStore",
    "CrateDBLoader",
    "CrateDBRetriever",
    "__version__",
]
