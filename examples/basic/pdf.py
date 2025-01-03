"""
Use CrateDB Vector Search with Sentence Transformers from Hugging Face.

- https://huggingface.co/sentence-transformers
- https://python.langchain.com/docs/integrations/text_embedding/sentence_transformers/

As input data, the example uses the canonical `state_of_the_union.txt`.

Synopsis::

    # Install prerequisites.
    pip install --upgrade langchain-huggingface langchain-cratedb langchain-text-splitters 'pypdf!=5.1.0'

    # Start database.
    docker run --rm -it --publish=4200:4200 crate/crate:nightly

    # Optionally set environment variable to configure CrateDB connection URL.
    export CRATEDB_SQLALCHEMY_URL="crate://crate@localhost/?schema=doc"

    # Run program.
    python examples/basic/pdf.py
"""  # noqa: E501
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "langchain-cratedb",
#   "langchain-huggingface",
#   "langchain-text-splitters",
#   "pypdf!=5.1.0",
# ]
# ///

import os
import typing as t

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_cratedb import CrateDBVectorStore

CRATEDB_SQLALCHEMY_URL = os.environ.get(
    "CRATEDB_SQLALCHEMY_URL", "crate://crate@localhost/?schema=testdrive"
)
RESOURCE_URL = "https://patentimages.storage.googleapis.com/1e/f5/93/346d19e0e43e92/EP0666666B1.pdf"


def get_documents() -> t.List[Document]:
    """
    Acquire data, return as LangChain documents.
    """

    # Define text splitter.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

    # Define resource loader.
    loader = PyPDFLoader(RESOURCE_URL)

    # Load PDF pages and split into fragments.
    fragments = []
    pages = loader.load()
    for page in pages:
        fragments += text_splitter.create_documents([page.page_content])
    return fragments


def main() -> None:
    # Set up LLM.
    # embeddings = OpenAIEmbeddings()  # noqa: ERA001
    # """
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
        model_name="all-MiniLM-L6-v2",  # noqa: ERA001
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
        # model_name="mixedbread-ai/mxbai-embed-xsmall-v1",  # noqa: ERA001
    )
    # """

    # Acquire documents.
    print("Acquiring data")
    documents = get_documents()

    # Embed each chunk, and load them into the vector store.
    print("Indexing data")
    vector_store = CrateDBVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        connection=CRATEDB_SQLALCHEMY_URL,
    )

    # Invoke a query, and display the first result.
    print("Querying data")
    queries = [
        "What is the invention about?",
        "What does the patent describe?",
        "Give me a summary, please.",
        "Which kind of system is it?",
        "Was ist das für ein System?",
        "De quel type de système s'agit-il?",
    ]
    for query in queries:
        print("=" * 42)
        print("Query:", query)
        print("=" * 42)
        docs = vector_store.similarity_search(query, k=3)
        for doc in docs:
            print(doc.page_content)
            print()
        print()

    vector_store.delete_collection()


if __name__ == "__main__":
    main()
