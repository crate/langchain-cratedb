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

from langchain_cratedb import CrateDBVectorStore

CRATEDB_SQLALCHEMY_URL = os.environ.get(
    "CRATEDB_SQLALCHEMY_URL", "crate://crate@localhost/?schema=testdrive"
)


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
    # Acquire documents.
    documents = get_documents()

    # Embed each chunk, and load them into the vector store.
    vector_store = CrateDBVectorStore.from_documents(
        documents, OpenAIEmbeddings(), connection=CRATEDB_SQLALCHEMY_URL
    )

    # Invoke a query, and display the first result.
    query = "What did the president say about Ketanji Brown Jackson"
    docs = vector_store.similarity_search(query)
    print(docs[0].page_content)


if __name__ == "__main__":
    main()
