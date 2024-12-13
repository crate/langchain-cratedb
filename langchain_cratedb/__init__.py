from importlib import metadata

from langchain_cratedb.chat_models import ChatCrateDB
from langchain_cratedb.document_loaders import CrateDBLoader
from langchain_cratedb.embeddings import CrateDBEmbeddings
from langchain_cratedb.retrievers import CrateDBRetriever
from langchain_cratedb.toolkits import CrateDBToolkit
from langchain_cratedb.tools import CrateDBTool
from langchain_cratedb.vectorstores import CrateDBVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatCrateDB",
    "CrateDBVectorStore",
    "CrateDBEmbeddings",
    "CrateDBLoader",
    "CrateDBRetriever",
    "CrateDBToolkit",
    "CrateDBTool",
    "__version__",
]
