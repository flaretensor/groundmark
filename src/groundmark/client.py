from collections.abc import Callable, Iterator
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Protocol

from groundmark.models import Progress, ResearchResult
from groundmark.research import Options, ResearchFailure, ResearchRunner


@dataclass(frozen=True)
class RunError:
    kind: str


ResearchEvent = Progress | ResearchResult | RunError


class ResearchClient(Protocol):
    def stream(self, question: str, options: Options | None = None) -> Iterator[ResearchEvent]: ...


def start_events(
    runner: ResearchRunner,
    question: str,
    options: Options | None = None,
    on_complete: Callable[[], None] | None = None,
) -> Iterator[ResearchEvent]:
    events: Queue[ResearchEvent | None] = Queue()

    def work():
        try:
            events.put(runner.run(question, options, events.put))
        except ResearchFailure as error:
            events.put(RunError(str(error)))
        except Exception:
            events.put(RunError("internal"))
        finally:
            if on_complete:
                on_complete()
            events.put(None)

    Thread(target=work, name="groundmark-research").start()

    def consume():
        while (item := events.get()) is not None:
            yield item

    return consume()


class LocalClient:
    def __init__(self, runner: ResearchRunner | None = None):
        self.runner = runner or ResearchRunner()

    def stream(self, question: str, options: Options | None = None) -> Iterator[ResearchEvent]:
        return start_events(self.runner, question, options)
