"""
Demonstrate chat history / conversational memory with CrateDB.

Synopsis::

    # Install prerequisites.
    pip install --upgrade langchain-cratedb

    # Start database.
    docker run --rm -it --publish=4200:4200 crate/crate:nightly

    # Optionally set environment variable to configure CrateDB connection URL.
    export CRATEDB_SQLALCHEMY_URL="crate://crate@localhost/?schema=doc"

    # Run program.
    python examples/basic/chat_history.py
"""  # noqa: E501
# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "langchain-cratedb",
# ]
# ///

import os
from pprint import pprint

from langchain_cratedb import CrateDBChatMessageHistory

CRATEDB_SQLALCHEMY_URL = os.environ.get(
    "CRATEDB_SQLALCHEMY_URL", "crate://crate@localhost/?schema=testdrive"
)


def main() -> None:
    chat_history = CrateDBChatMessageHistory(
        session_id="test_session",
        connection=CRATEDB_SQLALCHEMY_URL,
    )
    chat_history.add_user_message("Hello")
    chat_history.add_ai_message("Hi")
    pprint(chat_history.messages)


if __name__ == "__main__":
    main()
