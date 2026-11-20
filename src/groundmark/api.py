import json
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import Field, field_validator

from groundmark.client import RunError, start_events
from groundmark.models import Progress, ResearchResult
from groundmark.research import Options, ResearchFailure, ResearchRunner


class ResearchRequest(Options):
    question: str = Field(min_length=1, max_length=500)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question is empty")
        return value

    def options(self) -> Options:
        return Options.model_validate(self.model_dump(exclude={"question"}))


def _sse(events):
    for event in events:
        if isinstance(event, Progress):
            name, data = "progress", event.model_dump_json()
        elif isinstance(event, ResearchResult):
            name, data = "result", event.model_dump_json()
        elif isinstance(event, RunError):
            name, data = "error", json.dumps({"kind": event.kind})
        yield f"event: {name}\ndata: {data}\n\n"


def create_app(runner: ResearchRunner | None = None) -> FastAPI:
    app = FastAPI(title="Groundmark")
    active = Lock()
    runner = runner or ResearchRunner()

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        return JSONResponse({"detail": {"kind": "invalid_request"}}, status_code=422)

    def take_slot():
        if not active.acquire(blocking=False):
            raise HTTPException(status_code=503, detail={"kind": "busy"})

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/research", response_model=ResearchResult)
    def research(request: ResearchRequest):
        take_slot()
        try:
            return runner.run(request.question, request.options())
        except ResearchFailure as error:
            raise HTTPException(status_code=503, detail={"kind": str(error)}) from None
        except Exception:
            raise HTTPException(status_code=500, detail={"kind": "internal"}) from None
        finally:
            active.release()

    @app.post("/research/stream")
    def research_stream(request: ResearchRequest):
        take_slot()
        try:
            events = start_events(runner, request.question, request.options(), active.release)
        except RuntimeError:
            active.release()
            raise HTTPException(status_code=500, detail={"kind": "internal"}) from None
        return StreamingResponse(
            _sse(events),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()
