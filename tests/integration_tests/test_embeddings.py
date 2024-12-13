"""Test CrateDB embeddings."""

from typing import Type

from langchain_cratedb.embeddings import CrateDBEmbeddings
from langchain_tests.integration_tests import EmbeddingsIntegrationTests


class TestParrotLinkEmbeddingsIntegration(EmbeddingsIntegrationTests):
    @property
    def embeddings_class(self) -> Type[CrateDBEmbeddings]:
        return CrateDBEmbeddings

    @property
    def embedding_model_params(self) -> dict:
        return {"model": "nest-embed-001"}
