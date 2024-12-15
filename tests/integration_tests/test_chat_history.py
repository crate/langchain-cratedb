"""
Test cases for conversational memory from `langchain-postgres` and `langchain-mongodb`.
"""

import json
import uuid
from typing import Any, Generator, Tuple

import pytest
import sqlalchemy as sa
from langchain.memory import ConversationBufferMemory
from langchain.schema.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    _message_to_dict,
)
from langchain_community.chat_message_histories.sql import DefaultMessageConverter

from langchain_cratedb.chat_history import CrateDBChatMessageHistory


@pytest.fixture(autouse=True)
def reset_database(engine: sa.Engine) -> None:
    """
    Provision database with table schema and data.
    """
    with engine.connect() as connection:
        connection.execute(sa.text("DROP TABLE IF EXISTS test_table;"))
        connection.commit()


@pytest.fixture()
def sql_histories(
    engine: sa.Engine,
) -> Generator[Tuple[CrateDBChatMessageHistory, CrateDBChatMessageHistory], None, None]:
    """
    Provide the test cases with data fixtures.
    """
    message_history = CrateDBChatMessageHistory(
        session_id="123", connection=engine, table_name="test_table"
    )
    # Create history for other session
    other_history = CrateDBChatMessageHistory(
        session_id="456", connection=engine, table_name="test_table"
    )

    yield message_history, other_history
    message_history.clear()
    other_history.clear()


def test_add_messages(
    sql_histories: Tuple[CrateDBChatMessageHistory, CrateDBChatMessageHistory],
) -> None:
    history1, _ = sql_histories
    history1.add_user_message("Hello!")
    history1.add_ai_message("Hi there!")

    messages = history1.messages
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "Hello!"
    assert messages[1].content == "Hi there!"


def test_multiple_sessions(
    sql_histories: Tuple[CrateDBChatMessageHistory, CrateDBChatMessageHistory],
) -> None:
    history1, history2 = sql_histories

    # first session
    history1.add_user_message("Hello!")
    history1.add_ai_message("Hi there!")
    history1.add_user_message("Whats cracking?")

    # second session
    history2.add_user_message("Hellox")

    messages1 = history1.messages
    messages2 = history2.messages

    # Ensure the messages are added correctly in the first session
    assert len(messages1) == 3, "waat"
    assert messages1[0].content == "Hello!"
    assert messages1[1].content == "Hi there!"
    assert messages1[2].content == "Whats cracking?"

    assert len(messages2) == 1
    assert len(messages1) == 3
    assert messages2[0].content == "Hellox"
    assert messages1[0].content == "Hello!"
    assert messages1[1].content == "Hi there!"
    assert messages1[2].content == "Whats cracking?"


def test_clear_messages(
    sql_histories: Tuple[CrateDBChatMessageHistory, CrateDBChatMessageHistory],
) -> None:
    sql_history, other_history = sql_histories
    sql_history.add_user_message("Hello!")
    sql_history.add_ai_message("Hi there!")
    assert len(sql_history.messages) == 2
    # Now create another history with different session id
    other_history.add_user_message("Hellox")
    assert len(other_history.messages) == 1
    assert len(sql_history.messages) == 2
    # Now clear the first history
    sql_history.clear()
    assert len(sql_history.messages) == 0
    assert len(other_history.messages) == 1


def test_model_no_session_id_field_error(engine: sa.Engine) -> None:
    class Base(sa.orm.DeclarativeBase):
        pass

    class Model(Base):
        __tablename__ = "test_table"
        id = sa.Column(sa.Integer, primary_key=True)
        test_field = sa.Column(sa.Text)

    class CustomMessageConverter(DefaultMessageConverter):
        def get_sql_model_class(self) -> Any:
            return Model

    with pytest.raises(ValueError):
        CrateDBChatMessageHistory(
            "test",
            connection=engine,
            custom_message_converter=CustomMessageConverter("test_table"),
        )


def test_memory_with_message_store(engine: sa.Engine) -> None:
    """
    Test ConversationBufferMemory with a message store.
    """
    # Setup CrateDB as a message store.
    message_history = CrateDBChatMessageHistory(
        connection=engine, session_id="test-session"
    )
    memory = ConversationBufferMemory(
        memory_key="baz", chat_memory=message_history, return_messages=True
    )

    # Add a few messages.
    memory.chat_memory.add_ai_message("This is me, the AI")
    memory.chat_memory.add_user_message("This is me, the human")

    # Get the message history from the memory store and turn it into JSON.
    messages = memory.chat_memory.messages
    messages_json = json.dumps([_message_to_dict(msg) for msg in messages])

    # Verify the outcome.
    assert "This is me, the AI" in messages_json
    assert "This is me, the human" in messages_json

    # Clear the conversation history, and verify that.
    memory.chat_memory.clear()
    assert memory.chat_memory.messages == []


def test_sync_chat_history(engine: sa.Engine) -> None:
    table_name = "chat_history"
    session_id = str(uuid.UUID(int=123))

    chat_history = CrateDBChatMessageHistory(
        table_name=table_name, session_id=session_id, connection=engine
    )

    messages = chat_history.messages
    assert messages == []

    assert chat_history is not None

    # Get messages from the chat history
    messages = chat_history.messages
    assert messages == []

    chat_history.add_messages(
        [
            SystemMessage(content="Meow"),
            AIMessage(content="woof"),
            HumanMessage(content="bark"),
        ]
    )

    # Get messages from the chat history
    messages = chat_history.messages
    assert len(messages) == 3
    assert messages == [
        SystemMessage(content="Meow"),
        AIMessage(content="woof"),
        HumanMessage(content="bark"),
    ]

    chat_history.add_messages(
        [
            SystemMessage(content="Meow"),
            AIMessage(content="woof"),
            HumanMessage(content="bark"),
        ]
    )

    messages = chat_history.messages
    assert len(messages) == 6
    assert messages == [
        SystemMessage(content="Meow"),
        AIMessage(content="woof"),
        HumanMessage(content="bark"),
        SystemMessage(content="Meow"),
        AIMessage(content="woof"),
        HumanMessage(content="bark"),
    ]

    chat_history.clear()
    assert chat_history.messages == []
