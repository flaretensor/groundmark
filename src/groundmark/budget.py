import fcntl
import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Literal
from uuid import uuid4

NANOS_PER_USD = 1_000_000_000


class BudgetRefused(Exception):
    def __init__(
        self,
        message: str = "estimated provider budget reached",
        *,
        kind: Literal["credits", "usd", "calls"] = "usd",
    ):
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class Amount:
    credits: int = 0
    nanos: int = 0

    def __post_init__(self):
        if self.credits < 0 or self.nanos < 0:
            raise ValueError("Usage must be nonnegative")

    def __add__(self, other: "Amount") -> "Amount":
        return Amount(self.credits + other.credits, self.nanos + other.nanos)

    def exceeds(self, limit: "Amount") -> bool:
        return self.credits > limit.credits or self.nanos > limit.nanos


def usd_amount(credits: int, usd: float) -> Amount:
    return Amount(credits, int(Decimal(str(usd)) * NANOS_PER_USD))


class SpendLedger:
    def __init__(
        self,
        path: Path,
        total_limit: Amount,
        *,
        scope: str | None = None,
        scope_limit: Amount | None = None,
        scope_call_limit: int | None = None,
    ):
        self.path = path
        self.total_limit = total_limit
        if (scope is None) != (scope_limit is None):
            raise ValueError("Scope and scope limit must be specified together")
        if scope_call_limit is not None and (scope is None or scope_call_limit < 1):
            raise ValueError("A positive call limit requires a scope")
        self.scope = scope
        self.scope_limit = scope_limit
        self.scope_call_limit = scope_call_limit
        self.scope_blocked = False

    def _scoped_usage(self, rows: list[dict]) -> Amount:
        return self._used([row for row in rows if row.get("scope") == self.scope])

    def scope_total(self) -> Amount:
        with self._locked() as (rows, _):
            return self._scoped_usage(rows)

    def _scope_reservations(self, rows: list[dict]) -> int:
        return sum(row.get("scope") == self.scope and row["event"] == "reserve" for row in rows)

    def scope_reservations(self) -> int:
        with self._locked() as (rows, _):
            return self._scope_reservations(rows)

    def scope_run_ids(self) -> set[str]:
        with self._locked() as (rows, _):
            return {
                row["run_id"]
                for row in rows
                if row.get("scope") == self.scope and row["event"] == "reserve"
            }

    def scope_unsettled(self) -> int:
        with self._locked() as (rows, _):
            settled = {row["call_id"] for row in rows if row["event"] == "settle"}
            return sum(
                row.get("scope") == self.scope
                and row["event"] == "reserve"
                and row["call_id"] not in settled
                for row in rows
            )

    def _limits(self, rows: list[dict], run_id: str, limit: Amount):
        yield limit, self._used(rows, run_id), False
        yield self.total_limit, self._used(rows), self.scope is not None
        if self.scope_limit is not None:
            yield self.scope_limit, self._scoped_usage(rows), True

    def _refuse(self, kind, scoped: bool) -> None:
        self.scope_blocked |= scoped
        raise BudgetRefused(kind=kind)

    @contextmanager
    def _locked(self) -> Iterator[tuple[list[dict], object]]:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "r+", encoding="utf-8") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            try:
                try:
                    rows = [json.loads(line) for line in file]
                except (ValueError, UnicodeError):
                    raise BudgetRefused("unreadable spend ledger") from None
                yield rows, file
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)

    @staticmethod
    def _append(file, row: dict) -> None:
        file.write(json.dumps(row, separators=(",", ":")) + "\n")
        file.flush()
        os.fsync(file.fileno())

    @staticmethod
    def _used(rows: list[dict], run_id: str | None = None) -> Amount:
        charges = {}
        for row in rows:
            if row["event"] in {"reserve", "settle"} and (
                run_id is None or row["run_id"] == run_id
            ):
                charges[row["call_id"]] = Amount(row["credits"], row["nanos"])
        return sum(charges.values(), start=Amount())

    def total(self, run_id: str | None = None) -> Amount:
        with self._locked() as (rows, _):
            return self._used(rows, run_id)

    def remaining(self, run_id: str, limit: Amount) -> Amount:
        with self._locked() as (rows, _):
            limits = list(self._limits(rows, run_id, limit))
            return Amount(
                max(0, min(cap.credits - used.credits for cap, used, _ in limits)),
                max(0, min(cap.nanos - used.nanos for cap, used, _ in limits)),
            )

    def remaining_credits(self, run_id: str, limit: Amount) -> int:
        return self.remaining(run_id, limit).credits

    def start(self, run_id: str) -> None:
        with self._locked() as (rows, file):
            limits = [(self.total_limit, self._used(rows))]
            if self.scope_limit is not None:
                limits.append((self.scope_limit, self._scoped_usage(rows)))
            for limit, used in limits:
                if used.nanos >= limit.nanos:
                    self._refuse("usd", self.scope is not None)
                if used.credits >= limit.credits:
                    self._refuse("credits", self.scope is not None)
            self._append(file, {"event": "start", "run_id": run_id, "scope": self.scope})

    def reserve(self, run_id: str, operation: str, amount: Amount, limit: Amount) -> str:
        call_id = uuid4().hex
        with self._locked() as (rows, file):
            if self.scope_blocked:
                raise BudgetRefused("evaluation stage budget reached")
            if (
                self.scope_call_limit is not None
                and self._scope_reservations(rows) >= self.scope_call_limit
            ):
                self._refuse("calls", True)
            for ceiling, used, scoped in self._limits(rows, run_id, limit):
                after = used + amount
                if after.nanos > ceiling.nanos:
                    self._refuse("usd", scoped)
                if amount.credits and after.credits > ceiling.credits:
                    self._refuse("credits", scoped)
            self._append(
                file,
                {
                    "event": "reserve",
                    "run_id": run_id,
                    "call_id": call_id,
                    "operation": operation,
                    "scope": self.scope,
                    "credits": amount.credits,
                    "nanos": amount.nanos,
                },
            )
        return call_id

    def settle(self, run_id: str, call_id: str, amount: Amount) -> None:
        with self._locked() as (rows, file):
            original = next(
                (row for row in rows if row.get("call_id") == call_id and row["run_id"] == run_id),
                None,
            )
            if original is None:
                raise ValueError("Unknown reservation")
            if any(row.get("call_id") == call_id and row["event"] == "settle" for row in rows):
                return
            self._append(
                file,
                {
                    "event": "settle",
                    "run_id": run_id,
                    "call_id": call_id,
                    "scope": original.get("scope"),
                    "credits": amount.credits,
                    "nanos": amount.nanos,
                },
            )
