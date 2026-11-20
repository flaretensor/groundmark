from concurrent.futures import ThreadPoolExecutor

import pytest

from groundmark.budget import DEFAULT_RUN_LIMIT, Amount, BudgetRefused, SpendLedger


def test_reserve_then_settle_and_refuse_per_run(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.jsonl")
    call = ledger.reserve("run", "search", Amount(credits=5), DEFAULT_RUN_LIMIT)
    with pytest.raises(BudgetRefused):
        ledger.reserve("run", "search", Amount(credits=2), DEFAULT_RUN_LIMIT)
    ledger.settle("run", call, Amount(credits=1))
    assert ledger.total() == Amount(credits=1)
    assert ledger.remaining_credits("run", DEFAULT_RUN_LIMIT) == 5


def test_session_caps_and_run_limit(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.jsonl")
    for i in range(25):
        ledger.reserve(str(i), "search", Amount(6, 120_000_000), DEFAULT_RUN_LIMIT)
    for amount in (Amount(1), Amount(nanos=1)):
        with pytest.raises(BudgetRefused):
            ledger.reserve("extra", "probe", amount, DEFAULT_RUN_LIMIT)
    for i in range(12):
        ledger.start(str(i))
    with pytest.raises(BudgetRefused):
        ledger.start("extra")


def test_concurrent_reservations_share_capacity(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.jsonl")

    def reserve(_):
        try:
            ledger.reserve("run", "verify", Amount(nanos=80_000_000), DEFAULT_RUN_LIMIT)
        except BudgetRefused:
            return False
        return True

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(reserve, range(4))) == 1
