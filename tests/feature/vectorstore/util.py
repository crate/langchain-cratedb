import typing as t

import sqlalchemy as sa
from langchain_core.documents import Document

from langchain_cratedb.vectorstores.model import (
    COLLECTION_TABLE_NAME,
    EMBEDDING_TABLE_NAME,
)


def ensure_collection(session: sa.orm.Session, name: str) -> None:
    """
    Create the LangChain database table structure.

    TODO: Why is it done manually?
    """
    session.execute(
        sa.text(
            f"""
            CREATE TABLE IF NOT EXISTS {COLLECTION_TABLE_NAME} (
                uuid TEXT,
                name TEXT,
                cmetadata OBJECT
            );
            """
        )
    )
    session.execute(
        sa.text(
            f"""
            CREATE TABLE IF NOT EXISTS {EMBEDDING_TABLE_NAME} (
                id TEXT PRIMARY KEY,
                collection_id TEXT,
                embedding FLOAT_VECTOR(123),
                document TEXT,
                cmetadata OBJECT
            );
            """
        )
    )
    try:
        session.execute(
            sa.text(
                f"INSERT INTO {COLLECTION_TABLE_NAME} (uuid, name, cmetadata) "  # noqa: S608
                f"VALUES ('uuid-{name}', '{name}', {{}});"
            )
        )
        session.execute(sa.text(f"REFRESH TABLE {COLLECTION_TABLE_NAME}"))
    except sa.exc.IntegrityError:
        pass


def prune_document_ids(
    documents: t.Union[t.List[t.Tuple[Document, float]], t.List[Document]],
) -> None:
    for document in documents:
        if isinstance(document, tuple):
            document[0].id = None
        else:
            document.id = None  # type: ignore[union-attr]
