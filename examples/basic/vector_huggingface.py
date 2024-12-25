"""
Use CrateDB Vector Search with Sentence Transformers from Hugging Face.

- https://huggingface.co/sentence-transformers
- https://python.langchain.com/docs/integrations/text_embedding/sentence_transformers/

As input data, the example uses the canonical `state_of_the_union.txt`.

Synopsis::

    # Install prerequisites.
    pip install --upgrade langchain-cratedb langchain-huggingface

    # Start database.
    docker run --rm -it --publish=4200:4200 crate/crate:nightly

    # Optionally set environment variable to configure CrateDB connection URL.
    export CRATEDB_SQLALCHEMY_URL="crate://crate@localhost/?schema=doc"

    # Run program.
    python examples/basic/vector_huggingface.py
"""  # noqa: E501
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "langchain-huggingface",
#   "langchain-cratedb",
# ]
# ///

import os
import typing as t

import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=0)

    # Load a document, and split it into chunks.
    text = requests.get(RESOURCE_URL, timeout=10).text
    return text_splitter.create_documents([text])


def main() -> None:
    # Set up LLM.
    embeddings = HuggingFaceEmbeddings(
        # A small sentence-transformers model mapping sentences & paragraphs to a
        # 384 dimensional dense vector space and can be used for tasks like
        # clustering or semantic search.
        #
        # The model is intended to be used as a sentence and short paragraph encoder.
        # Given an input text, it outputs a vector which captures the semantic
        # information. The sentence vector may be used for information retrieval,
        # clustering or sentence similarity tasks.
        #
        # By default, input text longer than 256 word pieces is truncated.
        # https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        #
        # model_name="all-MiniLM-L6-v2",  # noqa: ERA001
        #
        #
        # Every Byte Matters: Introducing mxbai-embed-xsmall-v1
        # https://www.mixedbread.ai/blog/mxbai-embed-xsmall-v1
        #
        # An open-source English embedding model optimized for retrieval tasks developed
        # by Mixedbread. It is built upon `sentence-transformers/all-MiniLM-L6-v2` and
        # trained with the AnglE loss and Espresso.
        #
        # https://huggingface.co/mixedbread-ai/mxbai-embed-xsmall-v1
        #
        model_name="mixedbread-ai/mxbai-embed-xsmall-v1",
    )

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
