"""Test embedding model integration."""

from typing import Type

from langchain_cratedb.embeddings import CrateDBEmbeddings
from langchain_tests.unit_tests import EmbeddingsUnitTests


class TestParrotLinkEmbeddingsUnit(EmbeddingsUnitTests):
    @property
    def embeddings_class(self) -> Type[CrateDBEmbeddings]:
        return CrateDBEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
