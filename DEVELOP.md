# Development Documentation

## Set up sandbox
```shell
git clone https://github.com/crate/langchain-cratedb.git
cd langchain-cratedb
poetry install --with test,test_integration
```

## Run software tests
Run all tests.
```shell
make test
make integration_test
```
Run tests selectively.
```shell
pytest -vvv -k standard
```
```shell
pytest -vvv -k cratedb
```

## Genesis

This package has been bootstrapped using `langchain-cli integration new`,
see also [How to implement an integration package].

Other fragments have been derived from [langchain-datastax], [langchain-mongodb],
and [langchain-postgres].


[How to implement an integration package]: https://python.langchain.com/docs/contributing/how_to/integrations/package/
[langchain-datastax]: https://github.com/langchain-ai/langchain-datastax
[langchain-mongodb]: https://github.com/langchain-ai/langchain-mongodb
[langchain-postgres]: https://github.com/langchain-ai/langchain-postgres
