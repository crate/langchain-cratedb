# langchain-cratedb

[![Bluesky][badge-bluesky]][project-bluesky]
[![Release Notes][badge-release-notes]][project-release-notes]
[![CI][badge-ci]][project-ci]
[![Downloads per month][badge-downloads-per-month]][project-downloads]

[![Package version][badge-package-version]][project-pypi]
[![License][badge-license]][project-license]
[![Status][badge-status]][project-pypi]
[![Supported Python versions][badge-python-versions]][project-pypi]

» [Documentation]
| [Changelog]
| [PyPI]
| [Issues]
| [Source code]
| [License]
| [CrateDB]
| [Community Forum]

The `langchain-cratedb` package implements the [CrateDB provider for LangChain],
i.e. core LangChain abstractions using [CrateDB] or [CrateDB Cloud].

Feel free to use the abstractions as provided or else modify them / extend them
as appropriate for your own applications. We appreciate contributions of any kind.

## Introduction

CrateDB is a distributed and scalable SQL database for storing and analyzing
massive amounts of data in near real-time, even with complex queries.
It is PostgreSQL-compatible, and based on Lucene.

LangChain is a composable framework to build context-aware, reasoning
applications with large language models, leveraging your company’s data
and APIs.

LangChain for CrateDB is an AI/ML framework that unlocks the application
of LLM technologies to hands-on projects, covering many needs end-to-end.
It builds upon the large array of utilities bundled by the LangChain
toolkit and the ultra-fast indexing capabilities of CrateDB.

You can apply [LangChain] to implement text-based applications using commercial
models, for example provided by [OpenAI], or open-source models, for example
Meta's [Llama] multilingual text-only and text-image models.

## Installation

```bash
pip install --upgrade langchain-cratedb
```

## Requirements

The package currently supports CrateDB and its Python DB API driver,
available per [crate] package. It will be automatically installed
when installing the LangChain adapter.

You can run [CrateDB Self-Managed] or start using [CrateDB Cloud],
see [CrateDB Installation], or [CrateDB Cloud Console].

## Usage

To learn about the LangChain adapter for CrateDB, please refer to the
documentation and examples:

- [Using LangChain with CrateDB]
- [CrateDB LangChain examples]

### Vector Store

A few notebooks demonstrate how to use the CrateDB vector store functionality
around its `FLOAT_VECTOR` data type and its `KNN_MATCH` function together with
LangChain.

You will learn how to import and query unstructured data using the
`CrateDBVectorStore`, for example to create a retrieval augmented generation
(RAG) pipeline.

Retrieval-Augmented Generation (RAG) combines a retrieval system, which fetches
relevant documents, with a generative model, allowing it to incorporate external
knowledge for more accurate and informed responses.

- [Example: Basic vector search]
- [Example: Basic RAG]
- [Example: Advanced RAG with use case]

### Document Loader

This notebook demonstrates how to load documents from a CrateDB database, using
LangChain's `SQLDatabase` and `CrateDBLoader` interfaces, based on SQLAlchemy.

- [Example: Load data from database table]

### Chat History

The chat message history adapter helps to store and manage chat message history
in a CrateDB table, for supporting conversational memory.

- [Example: Chat message history]

### Full Cache

The standard / full cache avoids invoking the LLM when the supplied
prompt is exactly the same as one encountered already.

- [Example: CrateDBCache]

### Semantic Cache

The semantic cache allows users to retrieve cached prompts based on semantic
similarity between the user input and previously cached inputs, also avoiding
to invoke the LLM when not needed.

- [Example: CrateDBSemanticCache]


## Project Information

### Acknowledgements
Kudos to the authors of all the many software components this library is
inheriting from and building upon, most notably the [langchain-postgres]
package, and [langchain] itself.

### Contributing
The `langchain-cratedb` package is an open source project, and is
[managed on GitHub]. We appreciate contributions of any kind.

### License
The project uses the MIT license, like the langchain-postgres project
it is deriving from.


[CrateDB]: https://cratedb.com/database
[CrateDB Cloud]: https://cratedb.com/database/cloud
[CrateDB Cloud Console]: https://console.cratedb.cloud/
[CrateDB Installation]: https://cratedb.com/docs/guide/install/
[CrateDB LangChain examples]: https://github.com/crate/cratedb-examples/tree/main/topic/machine-learning/llm-langchain
[CrateDB provider for LangChain]: https://python.langchain.com/docs/integrations/providers/cratedb/
[CrateDB Self-Managed]: https://cratedb.com/database/self-managed
[CrateDBVectorStore]: https://github.com/crate/langchain-cratedb/blob/cratedb/docs/vectorstores.ipynb
[crate]: https://pypi.org/project/crate/
[Example: Advanced RAG with use case]: https://github.com/crate/cratedb-examples/blob/main/topic/machine-learning/llm-langchain/cratedb_rag_customer_support_langchain.ipynb
[Example: Chat message history]: https://github.com/crate/cratedb-examples/blob/main/topic/machine-learning/llm-langchain/conversational_memory.ipynb
[Example: CrateDBCache]: https://github.com/crate/langchain-cratedb/blob/main/examples/basic/cache.py
[Example: CrateDBSemanticCache]: https://github.com/crate/langchain-cratedb/blob/main/examples/basic/cache.py
[Example: Basic RAG]: https://github.com/crate/cratedb-examples/blob/main/topic/machine-learning/llm-langchain/cratedb_rag_customer_support.ipynb
[Example: Basic vector search]: https://github.com/crate/cratedb-examples/blob/main/topic/machine-learning/llm-langchain/vector_search.ipynb
[Example: Load data from database table]: https://github.com/crate/cratedb-examples/blob/main/topic/machine-learning/llm-langchain/document_loader.ipynb
[LangChain]: https://www.langchain.com/
[langchain]: https://github.com/langchain-ai/langchain
[langchain-postgres]: https://github.com/langchain-ai/langchain-postgres
[Llama]: https://www.llama.com/
[OpenAI]: https://openai.com/
[Using LangChain with CrateDB]: https://cratedb.com/docs/guide/integrate/langchain/

[Changelog]: https://github.com/crate/langchain-cratedb/blob/cratedb/CHANGES.md
[Community Forum]: https://community.cratedb.com/
[Documentation]: https://cratedb.com/docs/guide/integrate/langchain/
[Issues]: https://github.com/crate/langchain-cratedb/issues
[License]: https://github.com/crate/langchain-cratedb/blob/cratedb/LICENSE
[managed on GitHub]: https://github.com/crate/langchain-cratedb
[PyPI]: https://pypi.org/project/langchain-cratedb/
[Source code]: https://github.com/crate/langchain-cratedb

[badge-bluesky]: https://img.shields.io/badge/Bluesky-0285FF?logo=bluesky&logoColor=fff&label=Follow%20%40CrateDB
[badge-ci]: https://github.com/crate/langchain-cratedb/actions/workflows/ci.yml/badge.svg
[badge-downloads-per-month]: https://pepy.tech/badge/langchain-cratedb/month
[badge-license]: https://img.shields.io/github/license/crate/langchain-cratedb.svg
[badge-package-version]: https://img.shields.io/pypi/v/langchain-cratedb.svg
[badge-python-versions]: https://img.shields.io/pypi/pyversions/langchain-cratedb.svg
[badge-release-notes]: https://img.shields.io/github/release/crate/langchain-cratedb?label=Release+Notes
[badge-status]: https://img.shields.io/pypi/status/langchain-cratedb.svg
[project-bluesky]: https://bsky.app/search?q=cratedb
[project-ci]: https://github.com/crate/langchain-cratedb/actions/workflows/ci.yml
[project-downloads]: https://pepy.tech/project/langchain-cratedb/
[project-license]: https://github.com/crate/langchain-cratedb/blob/cratedb/LICENSE
[project-pypi]: https://pypi.org/project/langchain-cratedb
[project-release-notes]: https://github.com/crate/langchain-cratedb/releases
