import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.sqlite import SqliteSaver


@contextmanager
def build_checkpointer(url: str) -> Iterator[BaseCheckpointSaver]:
    if url.startswith(("postgresql://", "postgres://")):
        with PostgresSaver.from_conn_string(url) as saver:
            saver.setup()
            yield saver
        return

    path = Path(url)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.close(fd)
    with SqliteSaver.from_conn_string(str(path)) as saver:
        yield saver
