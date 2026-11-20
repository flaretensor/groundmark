from concurrent.futures import ThreadPoolExecutor

import pytest

from groundmark.budget import Amount, BudgetRefused, SpendLedger

RUN_LIMIT = Amount(6, 150_000_000)
TOTAL_LIMIT = Amount(150, 3_000_000_000)


def test_reserve_then_settle_and_refuse_per_run(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.jsonl", TOTAL_LIMIT)
    call = ledger.reserve("run", "search", Amount(credits=5), RUN_LIMIT)
    with pytest.raises(BudgetRefused) as refused:
        ledger.reserve("run", "search", Amount(credits=2), RUN_LIMIT)
    assert refused.value.kind == "credits"
    ledger.settle("run", call, Amount(credits=1))
    assert ledger.total() == Amount(credits=1)
    assert ledger.remaining_credits("run", RUN_LIMIT) == 5


def test_total_caps_without_run_count_limit(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.jsonl", TOTAL_LIMIT)
    for i in range(25):
        ledger.reserve(str(i), "search", Amount(6, 120_000_000), RUN_LIMIT)
    for amount, kind in ((Amount(1), "credits"), (Amount(nanos=1), "usd"), (Amount(1, 1), "usd")):
        with pytest.raises(BudgetRefused) as refused:
            ledger.reserve("extra", "probe", amount, RUN_LIMIT)
        assert refused.value.kind == kind
    with pytest.raises(BudgetRefused) as refused:
        ledger.start("extra")
    assert refused.value.kind == "usd"


def test_credit_overage_allows_usd_only_reservations(tmp_path):
    limit = Amount(credits=1, nanos=10)
    ledger = SpendLedger(tmp_path / "spend.jsonl", limit)
    call = ledger.reserve("run", "search", Amount(credits=1), limit)
    ledger.settle("run", call, Amount(credits=2))

    with pytest.raises(BudgetRefused) as refused:
        ledger.start("later")
    assert refused.value.kind == "credits"
    ledger.reserve("run", "synthesize", Amount(nanos=1), limit)
    assert ledger.total("run") == Amount(credits=2, nanos=1)


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
