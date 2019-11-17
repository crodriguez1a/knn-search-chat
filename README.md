# knn-search-chat

A super simple Q&amp;A chat-bot

This repository supports a tutorial that experiments with using semantic textual similarity and neighborhood search to map natural language questions to answers.

https://colab.research.google.com/drive/18UIXqyQZIhcR7e6f-DBdubQShFRPv51n

## Installation and Testing

Installation with `Pipenv`:
```
pipenv install
pipenv shell
```

Running Tests:
```
pytest
```

## Running the Server

Start the flask server:
`python -m server.server`

With the flask server running, run the following:
`python -m server.chat`

From the command prompt, begin typing a message using the command `say`:

```
(Cmd) say Hello
| Bot says: Hey
```
