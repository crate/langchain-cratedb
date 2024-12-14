# langchain-cratedb

[![Release Notes](https://img.shields.io/github/release/crate/langchain-cratedb)](https://github.com/crate/langchain-cratedb/releases)
[![CI](https://github.com/crate/langchain-cratedb/actions/workflows/ci.yml/badge.svg)](https://github.com/crate/langchain-cratedb/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Bluesky](https://img.shields.io/badge/Bluesky-0285FF?logo=bluesky&logoColor=fff&label=Follow%20%40CrateDB)](https://bsky.app/search?q=cratedb)

» [Documentation]
| [Changelog]
| [Community Forum]
| [PyPI]
| [Issues]
| [Source code]
| [License]
| [CrateDB]

The `langchain-cratedb` package implements core LangChain abstractions
using [CrateDB] or [CrateDB Cloud].

The package is released under the MIT license. 

Feel free to use the abstraction as provided or else modify them / extend them
as appropriate for your own application.

## Requirements

The package currently only supports the Python DB API driver, available per
[crate] package.

## Installation

```bash
pip install -U langchain-cratedb
```

## Usage

### ChatMessageHistory

The chat message history abstraction helps to persist chat message history
in a CrateDB table.

CrateDBChatMessageHistory is parameterized using a `table_name` and a `session_id`.

The `table_name` is the name of the table in the database where 
the chat messages will be stored.

The `session_id` is a unique identifier for the chat session. It can be assigned
by the caller using `uuid.uuid4()`.

```python
import uuid

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_cratedb import CrateDBChatMessageHistory

# Create the table schema (only needs to be done once)
dburi = "crate://crate@localhost:4200"
table_name = "chat_history"
session_id = str(uuid.uuid4())

# Initialize the chat history manager
chat_history = CrateDBChatMessageHistory(
    table_name,
    session_id,
    connection=dburi,
)

# Add messages to the chat history
chat_history.add_messages([
    SystemMessage(content="Meow"),
    AIMessage(content="woof"),
    HumanMessage(content="bark"),
])

print(chat_history.messages)
```


### VectorStore

See example notebook at [CrateDBVectorStore].


## Project Information

### Acknowledgements
Kudos to the authors of all the many software components this library is
inheriting from and building upon, most notably the [langchain-postgres]
package, and [langchain] itself.

### Contributing
The `langchain-cratedb` package is an open source project, and is
[managed on GitHub]. The project is still in its infancy, and
we appreciate contributions of any kind.

### License
The project uses the MIT license, like the langchain-postgres project
it is deriving from.


[Changelog]: https://github.com/crate/langchain-cratedb/blob/main/CHANGES.md
[Community Forum]: https://community.cratedb.com/
[crate]: https://pypi.org/project/crate/
[CrateDB]: https://cratedb.com/database
[CrateDB Cloud]: https://cratedb.com/database/cloud
[CrateDBVectorStore]: https://github.com/crate/langchain-cratedb/blob/main/docs/vectorstores.ipynb
[Documentation]: https://cratedb.com/docs/guide/integrate/langchain/
[Issues]: https://github.com/crate/langchain-cratedb/issues
[langchain]: https://github.com/langchain-ai/langchain
[langchain-postgres]: https://github.com/langchain-ai/langchain-postgres
[License]: https://github.com/crate/langchain-cratedb/blob/main/LICENSE
[managed on GitHub]: https://github.com/crate/langchain-cratedb
[PyPI]: https://pypi.org/project/langchain-cratedb/
[Source code]: https://github.com/crate/langchain-cratedb
