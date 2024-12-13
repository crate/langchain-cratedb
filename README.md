# langchain-cratedb

This package contains the LangChain integration with CrateDB

## Installation

```bash
pip install -U langchain-cratedb
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatCrateDB` class exposes chat models from CrateDB.

```python
from langchain_cratedb import ChatCrateDB

llm = ChatCrateDB()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`CrateDBEmbeddings` class exposes embeddings from CrateDB.

```python
from langchain_cratedb import CrateDBEmbeddings

embeddings = CrateDBEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`CrateDBLLM` class exposes LLMs from CrateDB.

```python
from langchain_cratedb import CrateDBLLM

llm = CrateDBLLM()
llm.invoke("The meaning of life is")
```
