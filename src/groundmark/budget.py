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
        kind: Literal["credits", "usd"] = "usd",
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
    def __init__(self, path: Path, total_limit: Amount):
        self.path = path
        self.total_limit = total_limit

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

    def remaining_credits(self, run_id: str, limit: Amount) -> int:
        with self._locked() as (rows, _):
            return max(
                0,
                min(
                    limit.credits - self._used(rows, run_id).credits,
                    self.total_limit.credits - self._used(rows).credits,
                ),
            )

    def start(self, run_id: str) -> None:
        with self._locked() as (rows, file):
            used = self._used(rows)
            if used.nanos >= self.total_limit.nanos:
                raise BudgetRefused("total provider budget reached")
            if used.credits >= self.total_limit.credits:
                raise BudgetRefused("total provider budget reached", kind="credits")
            self._append(file, {"event": "start", "run_id": run_id})

    def reserve(self, run_id: str, operation: str, amount: Amount, limit: Amount) -> str:
        call_id = uuid4().hex
        with self._locked() as (rows, file):
            run_after = self._used(rows, run_id) + amount
            total_after = self._used(rows) + amount
            if run_after.nanos > limit.nanos or total_after.nanos > self.total_limit.nanos:
                raise BudgetRefused()
            if amount.credits and (
                run_after.credits > limit.credits or total_after.credits > self.total_limit.credits
            ):
                raise BudgetRefused(kind="credits")
            self._append(
                file,
                {
                    "event": "reserve",
                    "run_id": run_id,
                    "call_id": call_id,
                    "operation": operation,
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
                    "credits": amount.credits,
                    "nanos": amount.nanos,
                },
            )
