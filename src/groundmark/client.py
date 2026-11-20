import json
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
    on_result: Callable[[ResearchResult], None] | None = None,
) -> Iterator[ResearchEvent]:
    events: Queue[ResearchEvent | None] = Queue()

    def work():
        try:
            result = runner.run(question, options, events.put)
            if on_result:
                on_result(result)
            events.put(result)
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


class HttpClient:
    def __init__(self, base_url: str, transport=None):
        self.base_url = base_url
        self.transport = transport

    def stream(self, question: str, options: Options | None = None) -> Iterator[ResearchEvent]:
        import httpx
        from pydantic import ValidationError

        payload = (options or Options()).model_dump(exclude_none=True)
        payload["question"] = question
        try:
            with (
                httpx.Client(
                    base_url=self.base_url,
                    timeout=httpx.Timeout(10, read=None),
                    transport=self.transport,
                    trust_env=False,
                ) as client,
                client.stream("POST", "/research/stream", json=payload) as response,
            ):
                if response.status_code == 422:
                    yield RunError("invalid_question")
                    return
                if response.status_code == 503:
                    response.read()
                    try:
                        kind = str(response.json()["detail"]["kind"])
                    except (ValueError, KeyError, TypeError):
                        kind = "busy"
                    yield RunError(kind)
                    return
                response.raise_for_status()
                for name, data in parse_sse(response.iter_lines()):
                    if name == "progress":
                        yield Progress.model_validate_json(data)
                    elif name == "result":
                        yield ResearchResult.model_validate_json(data)
                        return
                    elif name == "error":
                        error = RunError(**json.loads(data))
                        yield error
                        return
        except (httpx.HTTPError, ValidationError, ValueError, TypeError):
            pass
        yield RunError("invalid_response")


def parse_sse(lines: Iterator[str]) -> Iterator[tuple[str, str]]:
    name = ""
    data = []
    for line in lines:
        if not line:
            if name and data:
                yield name, "\n".join(data)
            name = ""
            data = []
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            name = value
        elif field == "data":
            data.append(value)
