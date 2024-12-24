# Development Documentation

## Set up sandbox
```shell
git clone https://github.com/crate/langchain-cratedb.git
cd langchain-cratedb
poetry install --with lint,test,typing
```

## Run CrateDB
```shell
docker run --rm \
  --publish=4200:4200 --publish=5432:5432 --env=CRATE_HEAP_SIZE=2g \
  crate:latest -Cdiscovery.type=single-node
```

## Validate codebase
Run linters and software tests.
```shell
make check
```
Format code.
```shell
make format
```

## Software tests
Run tests selectively.
```shell
pytest -vvv tests/test_docs.py tests/test_examples.py
```
```shell
pytest -vvv -k standard
```
```shell
pytest -vvv -k cratedb
```

## Release
```shell
poetry build
twine upload --skip-existing dist/*{.tar.gz,.whl}
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
