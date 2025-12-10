# LangChain CrateDB Adapter Changelog

## Unreleased
- Verified support for Python 3.14

## v0.2.0 - 2025-12-12
- Updated to langchain 1.x
- Updated to langchain-postgres 0.0.16

## v0.1.1 - 2025-02-05
- Updated to langchain-postgres 0.0.13

## v0.1.0 - 2025-01-03
- Added implementation and software tests for `CrateDBCache`,
  deriving from `SQLAlchemyCache`, and `CrateDBSemanticCache`,
  building upon `CrateDBVectorStore`.

## v0.0.0 - 2024-12-16
- Make it work
- Added implementations for `CrateDBVectorStore`, `CrateDBVectorStoreMultiCollection`,
  `CrateDBChatMessageHistory`, and `CrateDBLoader`.
