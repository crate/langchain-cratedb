import uuid
from typing import Any, List, Optional, Tuple

import sqlalchemy
from sqlalchemy.orm import Session, declarative_base, relationship

COLLECTION_TABLE_NAME = "langchain_collection"
EMBEDDING_TABLE_NAME = "langchain_embedding"


def generate_uuid() -> str:
    return str(uuid.uuid4())


class ModelFactory:
    """Provide SQLAlchemy model objects at runtime."""

    def __init__(self, dimensions: Optional[int] = None):
        from sqlalchemy_cratedb import FloatVector, ObjectType

        # While it does not have any function here, you will still need to supply a
        # dummy dimension size value for operations like deleting records.
        self.dimensions = dimensions or 1024

        Base: Any = declarative_base()

        # Optional: Use a custom schema for the langchain tables.
        # Base = declarative_base(metadata=MetaData(schema="langchain"))  # type: Any  # noqa: E501,ERA001

        class BaseModel(Base):
            """Base model for the SQL stores."""

            __abstract__ = True

        class CollectionStore(BaseModel):
            """Collection store."""

            __tablename__ = COLLECTION_TABLE_NAME
            __table_args__ = {"keep_existing": True}

            uuid = sqlalchemy.Column(
                # TODO: Does it also work like that with CrateDB?
                # UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
                sqlalchemy.String,
                primary_key=True,
                default=generate_uuid,
            )
            name = sqlalchemy.Column(sqlalchemy.String)
            cmetadata: sqlalchemy.Column = sqlalchemy.Column(ObjectType)

            embeddings = relationship(
                "EmbeddingStore",
                back_populates="collection",
                cascade="all, delete-orphan",
                passive_deletes=False,
            )

            @classmethod
            def get_by_name(
                cls, session: Session, name: str
            ) -> Optional["CollectionStore"]:
                return session.query(cls).filter(cls.name == name).first()  # type: ignore[attr-defined]

            @classmethod
            def get_by_names(
                cls, session: Session, names: List[str]
            ) -> List["CollectionStore"]:
                return session.query(cls).filter(cls.name.in_(names)).all()  # type: ignore[attr-defined]

            @classmethod
            def get_or_create(
                cls,
                session: Session,
                name: str,
                cmetadata: Optional[dict] = None,
            ) -> Tuple["CollectionStore", bool]:
                """
                Get or create a collection.
                Returns [Collection, bool] where the bool is True
                if the collection was created.
                """
                created = False
                collection = cls.get_by_name(session, name)
                if collection:
                    return collection, created

                collection = cls(name=name, cmetadata=cmetadata)
                session.add(collection)
                session.commit()
                created = True
                return collection, created

        class EmbeddingStore(BaseModel):
            """Embedding store."""

            __tablename__ = EMBEDDING_TABLE_NAME
            __table_args__ = {"keep_existing": True}

            id = sqlalchemy.Column(
                # Original: nullable=True, primary_key=True, index=True, unique=True
                sqlalchemy.String,
                nullable=False,
                primary_key=True,
                index=False,
                unique=False,
            )

            collection_id = sqlalchemy.Column(
                sqlalchemy.String,
                sqlalchemy.ForeignKey(
                    f"{CollectionStore.__tablename__}.uuid",
                    ondelete="CASCADE",
                ),
            )
            collection = relationship("CollectionStore", back_populates="embeddings")

            embedding: sqlalchemy.Column = sqlalchemy.Column(
                FloatVector(self.dimensions)
            )
            document: sqlalchemy.Column = sqlalchemy.Column(
                sqlalchemy.String, nullable=True
            )
            cmetadata: sqlalchemy.Column = sqlalchemy.Column(ObjectType, nullable=True)

        self.Base = Base
        self.BaseModel = BaseModel
        self.CollectionStore = CollectionStore
        self.EmbeddingStore = EmbeddingStore
