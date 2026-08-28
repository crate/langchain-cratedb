"""
Use CrateDB Vector Search with embeddings computed by a local Ollama server.

Ollama runs the embedding model in its own process and speaks HTTP, so this
program needs no account, no API key, and no machine learning stack of its
own. `vector_ollama.py` and `vector_openai.py` are otherwise the same program.

- https://ollama.com/library/nomic-embed-text
- https://python.langchain.com/docs/integrations/text_embedding/ollama/

As input data, the example uses the canonical `state_of_the_union.txt`.

Synopsis::

    # Install prerequisites.
    pip install --upgrade langchain-cratedb langchain-ollama langchain-text-splitters

    # Start database.
    docker run --rm -it --publish=4200:4200 crate/crate:nightly

    # Serve the embedding model. `nomic-embed-text` produces 768 dimensions,
    # well within the 2048 a CrateDB FLOAT_VECTOR column accepts.
    ollama serve
    ollama pull nomic-embed-text

    # Optionally set environment variables to configure the Ollama and CrateDB
    # endpoints.
    export OLLAMA_BASE_URL="http://localhost:11434"
    export CRATEDB_SQLALCHEMY_URL="crate://crate@localhost/?schema=doc"

    # Run program.
    python examples/basic/vector_ollama.py
"""  # noqa: E501
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "langchain-cratedb",
#   "langchain-ollama",
#   "langchain-text-splitters",
# ]
# ///

import os
import typing as t

import requests
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_cratedb import CrateDBVectorStore

CRATEDB_SQLALCHEMY_URL = os.environ.get(
    "CRATEDB_SQLALCHEMY_URL", "crate://crate@localhost/?schema=testdrive"
)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = "nomic-embed-text"


def get_documents() -> t.List[Document]:
    """
    Acquire data, return as LangChain documents.
    """

    # Define text splitter.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    # Load a document, and split it into chunks.
    url = "https://github.com/langchain-ai/langchain/raw/v0.0.325/docs/docs/modules/state_of_the_union.txt"
    text = requests.get(url, timeout=10).text
    return text_splitter.create_documents([text])


def main() -> None:
    # Set up the embedding model.
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    # Acquire documents.
    documents = get_documents()

    # Embed each chunk, and load them into the vector store.
    vector_store = CrateDBVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        connection=CRATEDB_SQLALCHEMY_URL,
    )

    # Invoke a query, and display the first result.
    query = "What did the president say about Ketanji Brown Jackson"
    docs = vector_store.similarity_search(query)
    print(docs[0].page_content)


if __name__ == "__main__":
    main()
