# LangChain CrateDB Adapter Changelog


## Unreleased
- Updated to `crate-2.0.0`, which uses `orjson` for JSON marshalling

## v0.1.0 - 2025-01-03
- Added implementation and software tests for `CrateDBCache`,
  deriving from `SQLAlchemyCache`, and `CrateDBSemanticCache`,
  building upon `CrateDBVectorStore`.

## v0.0.0 - 2024-12-16
- Make it work
- Added implementations for `CrateDBVectorStore`, `CrateDBVectorStoreMultiCollection`,
  `CrateDBChatMessageHistory`, and `CrateDBLoader`.
