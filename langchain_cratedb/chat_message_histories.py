import json
import typing as t

import sqlalchemy as sa
from langchain.schema import BaseMessage, _message_to_dict, messages_from_dict
from langchain_community.chat_message_histories.sql import (
    BaseMessageConverter,
    DBConnection,
    SQLChatMessageHistory,
)
from sqlalchemy_cratedb.support import refresh_after_dml, refresh_table


def create_message_model(table_name, DynamicBase):  # type: ignore
    """
    Create a message model for a given table name.

    This is a specialized version for CrateDB for generating integer-based
    primary keys using timestamps.

    Args:
        table_name: The name of the table to use.
        DynamicBase: The base class to use for the model.

    Returns:
        The model class.
    """

    # Model is declared inside a function to be able to use a dynamic table name.
    class Message(DynamicBase):
        __tablename__ = table_name
        id = sa.Column(sa.BigInteger, primary_key=True, server_default=sa.func.now())
        session_id = sa.Column(sa.Text)
        message = sa.Column(sa.Text)

    return Message


class CrateDBMessageConverter(BaseMessageConverter):
    """
    Convert BaseMessage to the SQLAlchemy model.
    The default message converter for CrateDBMessageConverter.

    It is the same as the generic `BaseMessageConverter` converter,
    but swaps in a different `create_message_model` function, that
    provides an SQLAlchemy model which uses an automatically-assigned
    primary key on the `id` column.
    """

    def __init__(self, table_name: str):
        self.model_class = create_message_model(table_name, sa.orm.declarative_base())

    def from_sql_model(self, sql_message: t.Any) -> BaseMessage:
        return messages_from_dict([json.loads(sql_message.message)])[0]

    def to_sql_model(self, message: BaseMessage, session_id: str) -> t.Any:
        return self.model_class(
            session_id=session_id, message=json.dumps(_message_to_dict(message))
        )

    def get_sql_model_class(self) -> t.Any:
        return self.model_class


class CrateDBChatMessageHistory(SQLChatMessageHistory):
    """
    Chat message history stored in an SQL database.

    It is the same as the generic `SQLChatMessageHistory` implementation,
    but swaps in a different message converter by default.
    """

    def __init__(
        self,
        session_id: str,
        connection_string: t.Optional[str] = None,
        table_name: str = "message_store",
        session_id_field_name: str = "session_id",
        custom_message_converter: t.Optional[BaseMessageConverter] = None,
        connection: t.Union[None, DBConnection] = None,
        engine_args: t.Optional[t.Dict[str, t.Any]] = None,
        async_mode: t.Optional[bool] = None,  # Use only if connection is a string
    ):
        custom_message_converter = custom_message_converter or CrateDBMessageConverter(
            table_name
        )

        super().__init__(
            session_id,
            connection_string=connection_string,
            table_name=table_name,
            session_id_field_name=session_id_field_name,
            custom_message_converter=custom_message_converter,
            connection=connection,
            engine_args=engine_args,
            async_mode=async_mode,
        )

        # Patch dialect to invoke `REFRESH TABLE` after each DML operation.
        refresh_after_dml(self.Session)

    def clear(self) -> None:
        """
        Needed for CrateDB to synchronize data because `on_flush` did not catch it.
        """
        outcome = super().clear()
        with self.Session() as session:
            refresh_table(session, self.sql_model_class)
        return outcome
