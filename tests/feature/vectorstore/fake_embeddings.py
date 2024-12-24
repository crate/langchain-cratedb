from typing import Dict, List

from langchain_community.embeddings import FakeEmbeddings

ADA_TOKEN_COUNT = 1536


class FakeEmbeddingsWithAdaDimension(FakeEmbeddings):
    """Fake embeddings functionality for testing."""

    size: int = ADA_TOKEN_COUNT
    """The size of the embedding vector."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings."""
        return [
            [float(1.0)] * (ADA_TOKEN_COUNT - 1) + [float(i)] for i in range(len(texts))
        ]

    def embed_query(self, text: str) -> List[float]:
        """Return simple embeddings."""
        return [float(1.0)] * (ADA_TOKEN_COUNT - 1) + [float(0.0)]


class ConsistentFakeEmbeddingsWithAdaDimension(FakeEmbeddingsWithAdaDimension):
    """
    Fake embeddings which remember all the texts seen so far to return
    consistent vectors for the same texts.

    Other than this, they also have a fixed dimensionality, which is
    important in this case.
    """

    def __init__(self, *args: List, **kwargs: Dict) -> None:
        super().__init__(size=ADA_TOKEN_COUNT)
