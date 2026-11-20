from concurrent.futures import ThreadPoolExecutor

import pytest

from groundmark.budget import Amount, BudgetRefused, SpendLedger

RUN_LIMIT = Amount(6, 150_000_000)
TOTAL_LIMIT = Amount(150, 3_000_000_000)


def test_reserve_then_settle_and_refuse_per_run(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.jsonl", TOTAL_LIMIT)
    call = ledger.reserve("run", "search", Amount(credits=5), RUN_LIMIT)
    with pytest.raises(BudgetRefused):
        ledger.reserve("run", "search", Amount(credits=2), RUN_LIMIT)
    ledger.settle("run", call, Amount(credits=1))
    assert ledger.total() == Amount(credits=1)
    assert ledger.remaining_credits("run", RUN_LIMIT) == 5


def test_total_caps_without_run_count_limit(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.jsonl", TOTAL_LIMIT)
    for i in range(25):
        ledger.reserve(str(i), "search", Amount(6, 120_000_000), RUN_LIMIT)
    for amount in (Amount(1), Amount(nanos=1)):
        with pytest.raises(BudgetRefused):
            ledger.reserve("extra", "probe", amount, RUN_LIMIT)
    with pytest.raises(BudgetRefused):
        ledger.start("extra")


def test_start_has_no_run_count_limit(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.jsonl", TOTAL_LIMIT)
    for i in range(50):
        ledger.start(str(i))


def test_concurrent_reservations_share_capacity(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.jsonl", TOTAL_LIMIT)

    def reserve(_):
        try:
            ledger.reserve("run", "verify", Amount(nanos=80_000_000), RUN_LIMIT)
        except BudgetRefused:
            return False
        return True

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(reserve, range(4))) == 1
