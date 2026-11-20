from time import perf_counter_ns

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from groundmark.models import SpanRecord


class RunTelemetry:
    def __init__(self, endpoint: str = "", headers: dict[str, str] | None = None):
        self.provider = TracerProvider(
            resource=Resource.create({"service.name": "groundmark"}), shutdown_on_exit=False
        )
        self.memory = InMemorySpanExporter()
        self.provider.add_span_processor(SimpleSpanProcessor(self.memory))
        if endpoint:
            endpoint = endpoint.rstrip("/")
            if not endpoint.endswith("/v1/traces"):
                endpoint += "/v1/traces"
            self.provider.add_span_processor(
                BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, headers=headers, timeout=3))
            )
        self.tracer = self.provider.get_tracer("groundmark")
        self.started = perf_counter_ns()

    def span(self, name: str, **attributes):
        return self.tracer.start_as_current_span(
            name, attributes=attributes, record_exception=False, set_status_on_exception=False
        )

    def collect(self) -> tuple[float, list[SpanRecord]]:
        elapsed = (perf_counter_ns() - self.started) / 1_000_000
        spans = [
            SpanRecord(
                name=span.name,
                trace_id=f"{span.context.trace_id:032x}",
                span_id=f"{span.context.span_id:016x}",
                parent_id=f"{span.parent.span_id:016x}" if span.parent else None,
                duration_ms=(span.end_time - span.start_time) / 1_000_000,
                attributes=dict(span.attributes),
            )
            for span in self.memory.get_finished_spans()
        ]
        return elapsed, spans

    def close(self) -> None:
        self.provider.shutdown()
