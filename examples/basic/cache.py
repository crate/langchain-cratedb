# ruff: noqa: T201
"""
Use CrateDB to cache LLM prompts and responses.

The standard / full cache avoids invoking the LLM when the supplied
prompt is exactly the same as one encountered already.

The semantic cache allows users to retrieve cached prompts based on semantic
similarity between the user input and previously cached inputs.

When turning on the cache, redundant LLM conversations don't need
to talk to the LLM (API), so they can also work offline.
"""

import os

import sqlalchemy as sa
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from langchain_cratedb import CrateDBCache, CrateDBSemanticCache

"""
Prerequisites: Because this program uses OpenAI's APIs, you need to supply an
authentication token. Either set the environment variable `OPENAI_API_KEY`,
or optionally configure your token here after enabling the code fragment.
"""
# _ = os.environ.setdefault(
#    "OPENAI_API_KEY", "sk-XJZ7pfog5Gp8Kus8D--invalid--0CJ5lyAKSefZLaV1Y9S1"
# )


def standard_cache() -> None:
    """
    Demonstrate LangChain standard cache with CrateDB.
    """

    # Configure cache.
    engine = sa.create_engine("crate://crate@localhost:4200/?schema=testdrive")
    set_llm_cache(CrateDBCache(engine))

    # Invoke LLM conversation.
    llm = ChatOpenAI(model_name="gpt-6-luna")  # type: ignore[call-arg]
    print()
    print("Asking with standard cache:")
    answer = llm.invoke("What is the answer to everything?")
    print(answer.content)

    # Turn off cache.
    set_llm_cache(None)


def semantic_cache() -> None:
    """
    Demonstrate LangChain semantic cache with CrateDB.
    """

    # Configure LLM models.
    model_name_embedding = "text-embedding-3-small"

    model_name_chat = "gpt-6-luna"

    # Configure embeddings.
    embeddings = OpenAIEmbeddings(model=model_name_embedding)

    # Configure cache.
    engine = sa.create_engine("crate://crate@localhost:4200/?schema=testdrive")
    set_llm_cache(
        CrateDBSemanticCache(
            embedding=embeddings,
            connection=engine,
            search_threshold=1.0,
        )
    )

    # Invoke LLM conversation.
    llm = ChatOpenAI(
        model_name=model_name_chat,  # type: ignore[call-arg]
    )
    print()
    print("Asking with semantic cache:")
    answer = llm.invoke("What is the answer to everything?")
    print(answer.content)

    # Turn off cache.
    set_llm_cache(None)


def main() -> None:
    standard_cache()
    semantic_cache()


if __name__ == "__main__":
    main()
