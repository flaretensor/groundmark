from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from prometheus_client import CollectorRegistry, generate_latest

from groundmark.analytics import run_status
from groundmark.models import ResearchResult


class RunMetrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        reader = PrometheusMetricReader(registry=self.registry)
        self.provider = MeterProvider(metric_readers=[reader], shutdown_on_exit=False)
        meter = self.provider.get_meter("groundmark")
        self.runs = meter.create_counter("groundmark.runs")
        self.latency = meter.create_histogram(
            "groundmark.run.latency",
            unit="s",
            explicit_bucket_boundaries_advisory=[1, 3, 5, 10, 15, 30, 45, 60, 90, 120, 180, 300],
        )
        self.credits = meter.create_counter("groundmark.tavily.credits")
        self.spend = meter.create_counter("groundmark.nebius.estimated_usd")
        self.claims = meter.create_counter("groundmark.claim.statuses")
        self.errors = meter.create_counter("groundmark.provider.errors")

    def record(self, result: ResearchResult) -> None:
        trace = result.trace
        labels = {"status": run_status(result)}
        self.runs.add(1, labels)
        self.latency.record(trace.total_latency_ms / 1000, labels)
        self.credits.add(trace.tavily_credits)
        self.spend.add(trace.estimated_usd)
        for status, count in trace.claim_status_counts.items():
            self.claims.add(count, {"status": status})
        for error in trace.errors:
            if error.provider:
                self.errors.add(1, {"provider": error.provider, "kind": error.kind})

    def render(self) -> bytes:
        return generate_latest(self.registry)
