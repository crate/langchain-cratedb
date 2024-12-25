"""
Use CrateDB Vector Search with OpenAI embeddings.

As input data, the example uses the canonical `state_of_the_union.txt`.

Synopsis::

    # Install prerequisites.
    pip install --upgrade langchain-cratedb langchain-openai

    # Start database.
    docker run --rm -it --publish=4200:4200 crate/crate:nightly

    # Configure: Set environment variables to configure OpenAI authentication token
    # and optionally CrateDB connection URL.
    export OPENAI_API_KEY="<API KEY>"
    export CRATEDB_SQLALCHEMY_URL="crate://crate@localhost/?schema=doc"

    # Run program.
    python examples/basic/vector_search.py
"""  # noqa: E501
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "langchain-openai",
#   "langchain-cratedb",
# ]
# ///

import os
import typing as t

import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import (
    ExperimentalMarkdownSyntaxTextSplitter,
    MarkdownTextSplitter,
)

from langchain_cratedb import CrateDBVectorStore

CRATEDB_SQLALCHEMY_URL = os.environ.get(
    "CRATEDB_SQLALCHEMY_URL", "crate://crate@localhost/?schema=testdrive"
)
# TODO: Change URL to repository after merging.
RESOURCE_URL = "https://gist.github.com/amotl/a5dd9814d1865b14248ca97eb8075f96/raw/Universal_Declaration_of_Human_Rights.md"


def get_documents() -> t.List[Document]:
    """
    Acquire data, return as LangChain documents.
    """

    # Define text splitter.
    text_splitter = MarkdownTextSplitter(chunk_size=350, chunk_overlap=0)

    # Load a document, and split it into chunks.
    text = requests.get(RESOURCE_URL, timeout=10).text
    return text_splitter.create_documents([text])


def main() -> None:
    # Set up LLM.
    embeddings = OpenAIEmbeddings()

    # Acquire documents.
    documents = get_documents()

    # Embed each chunk, and load them into the vector store.
    vector_store = CrateDBVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        connection=CRATEDB_SQLALCHEMY_URL,
    )

    # Invoke a query, and display the first result.
    query = "What does the declaration say about freedom?"
    docs = vector_store.similarity_search(query)
    for doc in docs:
        print("=" * 42)
        print(doc.page_content)


if __name__ == "__main__":
    main()
