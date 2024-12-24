# Troubleshooting

This page enumerates a few common problems and errors, accompanied by
recommendations how to resolve them.

## `OPENAI_API_KEY` missing
When using OpenAI's APIs, you need to supply an authentication token.
Otherwise, you may observe such an error message.
```
OpenAIError: The api_key client option must be set either by passing
api_key to the client or by setting the OPENAI_API_KEY environment variable
```

## `OPENAI_API_KEY` invalid
If you are observing an error message like this, an OpenAI API token
is supplied, but might be wrong or invalid.
```
AuthenticationError: Error code: 401 - {'error': {'message':
'Incorrect API key provided: sk-XJZ7p***************************************Y9S1.
You can find your API key at https://platform.openai.com/account/api-keys.',
'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}
```

## Using multiple vector sizes in parallel
LangChain uses two database tables, `langchain_collection` and `langchain_embedding`.
Because `langchain_embedding` includes a column definition `embedding FLOAT_VECTOR(1536)`,
and for CrateDB, a fixed vector size is obligatory, you need to use different
database schemas. You can define a database schema within the SQLAlchemy
connection URL.
```python
import sqlalchemy as sa
engine = sa.create_engine("crate://crate@localhost:4200/?schema=vector1536")
engine = sa.create_engine("crate://crate@localhost:4200/?schema=vector2048")
```
