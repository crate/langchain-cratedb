# Backlog

## Iteration +1
- Release 0.0.0
- Revise "Project links" (see PyPI)
- Mitigate `CRATEDB_API_KEY`
- Provide async variants for all methods
- Release 0.0.1
- Add CrateDB adapter to LangChain's "Provider" list 
- Documentation: Update all Notebooks to provide concise usage information
- Release 0.1.0

## Iteration +2
- Documentation: Improve `as_retriever` feature:
  - https://python.langchain.com/docs/tutorials/retrievers/#retrievers
  - https://python.langchain.com/api_reference/mongodb/vectorstores/langchain_mongodb.vectorstores.MongoDBAtlasVectorSearch.html

## Iteration +3
- Document retrieval demo (PDF)
  - Import tests from `langchain-mongodb` about PDFs
  - https://github.com/langchain-ai/langchain-mongodb/blob/main/libs/mongodb/tests/integration_tests/conftest.py
  - https://github.com/microsoft/markitdown (https://api.python.langchain.com/en/latest/_modules/langchain_community/document_loaders/markdown.html)
  - https://github.com/getomni-ai/zerox (https://python.langchain.com/docs/integrations/document_loaders/zeroxpdfloader/#loader-features)
  - https://github.com/freedmand/textra
  - https://github.com/gsidhu/winocr_cli
- Unlock other subsystems: Cache, Pipeline, Docstores, Retrievers, Store, ByteStore
  - https://python.langchain.com/api_reference/astradb/index.html
  - https://python.langchain.com/api_reference/elasticsearch/index.html
  - https://github.com/langchain-ai/langchain-mongodb/tree/main/libs/mongodb/langchain_mongodb
  - https://python.langchain.com/api_reference/couchbase/cache/langchain_couchbase.cache.CouchbaseSemanticCache.html
  - https://python.langchain.com/api_reference/mongodb/cache/langchain_mongodb.cache.MongoDBAtlasSemanticCache.html
  - https://python.langchain.com/api_reference/community/storage/langchain_community.storage.sql.SQLStore.html
  - https://python.langchain.com/api_reference/community/storage/langchain_community.storage.mongodb.MongoDBStore.html
  - https://python.langchain.com/api_reference/community/storage/langchain_community.storage.mongodb.MongoDBByteStore.html
- SQL-based Docstores
  - https://github.com/langchain-ai/langchain/discussions/12085
  - https://stackoverflow.com/questions/77438251/langchain-parentdocumetretriever-save-and-load
  - SQLStrStore and SQLDocStore persist `str` and `Document` objects, but they can be extended easily.
    Thanks, @gcheron.
    - https://github.com/langchain-ai/langchain/pull/15909
    - https://github.com/gcheron/langchain/blob/sql-store/docs/docs/integrations/stores/sql.ipynb
  - https://medium.com/@guilhem.cheron35/sql-storage-langchain-rags-inmemorystore-alternative-ex-with-parentdocumentretriever-pgvector-5cc162950d77
- Get into / do more with Retrievers
  - https://python.langchain.com/docs/concepts/retrieval/
  - https://api.python.langchain.com/en/latest/community/retrievers.html
  - https://api.python.langchain.com/en/latest/community/retrievers/langchain_community.retrievers.bm25.BM25Retriever.html#langchain_community.retrievers.bm25.BM25Retriever
  - https://api.python.langchain.com/en/latest/community/retrievers/langchain_community.retrievers.elastic_search_bm25.ElasticSearchBM25Retriever.html#langchain_community.retrievers.elastic_search_bm25.ElasticSearchBM25Retriever
  - https://api.python.langchain.com/en/latest/community/retrievers/langchain_community.retrievers.qdrant_sparse_vector_retriever.QdrantSparseVectorRetriever.html
  - https://api.python.langchain.com/en/latest/community/retrievers/langchain_community.retrievers.weaviate_hybrid_search.WeaviateHybridSearchRetriever.html
  - https://python.langchain.com/docs/how_to/time_weighted_vectorstore/
- Translators
  - https://python.langchain.com/docs/how_to/query_constructing_filters/
- Misc
  - https://python.langchain.com/api_reference/experimental/sql/langchain_experimental.sql.base.SQLDatabaseChain.html#langchain_experimental.sql.base.SQLDatabaseChain
  - https://python.langchain.com/api_reference/experimental/tabular_synthetic_data/langchain_experimental.tabular_synthetic_data.base.SyntheticDataGenerator.html#langchain_experimental.tabular_synthetic_data.base.SyntheticDataGenerator
  - https://python.langchain.com/api_reference/experimental/sql/langchain_experimental.sql.base.SQLDatabaseChain.html#langchain_experimental.sql.base.SQLDatabaseChain
  - https://python.langchain.com/api_reference/experimental/retrievers/langchain_experimental.retrievers.vector_sql_database.VectorSQLDatabaseChainRetriever.html#langchain_experimental.retrievers.vector_sql_database.VectorSQLDatabaseChainRetriever
- Product
  - https://aws.amazon.com/kendra/
  - https://aws.amazon.com/bedrock/knowledge-bases/
  - https://cloud.google.com/document-ai
  - https://cloud.google.com/generative-ai-app-builder/docs/try-enterprise-search
  - https://cloud.google.com/generative-ai-app-builder/docs/try-media-search
  - https://python.langchain.com/docs/tutorials/retrievers/
  - https://api.python.langchain.com/en/latest/community/retrievers/langchain_community.retrievers.bedrock.AmazonKnowledgeBasesRetriever.html#langchain_community.retrievers.bedrock.AmazonKnowledgeBasesRetriever
  - https://api.python.langchain.com/en/latest/community/retrievers/langchain_community.retrievers.google_vertex_ai_search.GoogleVertexAISearchRetriever.html
  - https://api.python.langchain.com/en/latest/community/retrievers/langchain_community.retrievers.google_cloud_documentai_warehouse.GoogleDocumentAIWarehouseRetriever.html

## Done
- Make it work
- Remove JSONB, only PostgreSQL has it
- Update README: https://github.com/langchain-ai/langchain-postgres
- Prune subsystems that are not applicable
- Rename `chat_message_histories` to just `chat_history`, like ES is doing it
- Minimally update documentation for uploading to PyPI v0.0.0
- Add code coverage reporting
- Dependencies: Use version ranges, focused around upper bounds
- Add PyPI project metadata
