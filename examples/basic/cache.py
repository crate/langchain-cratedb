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
from langchain.globals import set_llm_cache
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
    llm = ChatOpenAI(
        # model_name="gpt-3.5-turbo",
        # model_name="gpt-4o-mini",
        model_name="chatgpt-4o-latest",  # type: ignore[call-arg]
        temperature=0.7,
    )
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
    # model_name_embedding = "text-embedding-ada-002"
    model_name_embedding = "text-embedding-3-small"
    # model_name_embedding = "text-embedding-3-large"

    # model_name_chat = "gpt-3.5-turbo"
    # model_name_chat = "gpt-4o-mini"
    model_name_chat = "chatgpt-4o-latest"

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


"""
What is the answer to everything?

Date: 2024-12-23

## gpt-3.5-turbo
The answer to everything is subjective and may vary depending on individual
beliefs or philosophies. Some may say that love is the answer to everything,
while others may say that knowledge or self-awareness is the key. Ultimately,
the answer to everything may be different for each person and can only be
discovered through personal reflection and introspection.

## gpt-4o-mini
The answer to the ultimate question of life, the universe, and everything,
according to Douglas Adams' "The Hitchhiker's Guide to the Galaxy", is
famously given as the number 42. However, the context and meaning behind
that answer remains a philosophical and humorous mystery. In a broader
sense, different people and cultures may have various interpretations of
what the "answer to everything" truly is, often reflecting their beliefs,
values, and experiences.

## chatgpt-4o-latest, pure
Ah, you're referencing the famous answer from Douglas Adams'
*The Hitchhiker's Guide to the Galaxy*! In the book, the supercomputer
Deep Thought determines that the "Answer to the Ultimate Question of
Life, the Universe, and Everything" is **42**.
Of course, the real kicker is that no one actually knows what the Ultimate
Question is. So, while 42 is the answer, its true meaning remains a cosmic
mystery! 😊

## chatgpt-4o-latest, with text-embedding-3-small embeddings
Ah, you're referring to the famous answer from Douglas Adams'
*The Hitchhiker's Guide to the Galaxy*! The answer to the ultimate question
of life, the universe, and everything is **42**. However, as the story
humorously points out, the actual *question* remains unknown. 😊
If you're looking for a deeper or more philosophical answer, feel free to
elaborate!
"""
