"""
Use CrateDB Vector Search to answer questions about a PDF document.

As input data, the example uses the European patent EP0666666B1, loaded
straight from its canonical URL. The specification is published in English,
German, and French, so the same question can be asked in any of them.

- https://patents.google.com/patent/EP0666666B1/
- https://python.langchain.com/docs/integrations/document_loaders/pypdfloader/

Synopsis::

    # Install prerequisites.
    pip install --upgrade langchain-cratedb langchain-community langchain-openai langchain-text-splitters pypdf

    # Start database.
    docker run --rm -it --publish=4200:4200 crate/crate:nightly

    # Configure: Set environment variables to configure OpenAI authentication token
    # and optionally CrateDB connection URL.
    export OPENAI_API_KEY="<API KEY>"
    export CRATEDB_SQLALCHEMY_URL="crate://crate@localhost/?schema=doc"

    # Run program.
    python examples/basic/pdf.py
"""  # noqa: E501
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "langchain-cratedb",
#   "langchain-community",
#   "langchain-openai",
#   "langchain-text-splitters",
#   "pypdf",
# ]
# ///

import os
import typing as t

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_cratedb import CrateDBVectorStore

CRATEDB_SQLALCHEMY_URL = os.environ.get(
    "CRATEDB_SQLALCHEMY_URL", "crate://crate@localhost/?schema=testdrive"
)
RESOURCE_URL = "https://patentimages.storage.googleapis.com/1e/f5/93/346d19e0e43e92/EP0666666B1.pdf"


def get_documents() -> t.List[Document]:
    """
    Acquire the PDF, return its pages as LangChain documents.
    """

    # Define resource loader.
    loader = PyPDFLoader(RESOURCE_URL)

    # Define text splitter.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

    # Load PDF pages, and split each one into fragments.
    fragments = []
    for page in loader.load():
        fragments += text_splitter.create_documents([page.page_content])
    return fragments


def main() -> None:
    # Acquire documents.
    documents = get_documents()

    # Embed each fragment, and load them into the vector store.
    vector_store = CrateDBVectorStore.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(),
        connection=CRATEDB_SQLALCHEMY_URL,
    )

    # Ask the same question in each language the specification is published in.
    queries = [
        "What is the invention about?",
        "Was ist das für ein System?",
        "De quel type de système s'agit-il?",
    ]
    for query in queries:
        print("=" * 42)
        print("Query:", query)
        print("=" * 42)
        for doc in vector_store.similarity_search(query, k=2):
            print(doc.page_content)
            print()


if __name__ == "__main__":
    main()
