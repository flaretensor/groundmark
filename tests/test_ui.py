from pathlib import Path

from streamlit.testing.v1 import AppTest

from groundmark.models import ConflictResult, Progress
from groundmark.research import EXAMPLES

UI = Path("src/groundmark/ui.py").resolve()


def test_screen_renders_claims_sources_and_trace(monkeypatch, outcome):
    outcome.missing_information = ["Costs of $5 and $6 | tax."]
    outcome.sources.append(
        outcome.sources[0].model_copy(update={"source_id": "S2", "title": "Unread source"})
    )
    outcome.conflicts.append(
        ConflictResult(
            topic="Opening date",
            description="S1 and [S2] give different dates.",
            source_ids=["S1"],
            origin="synthesis",
        )
    )
    outcome.conflicts.append(
        ConflictResult(
            topic="Tokyo has 14 million residents.",
            description="Tokyo has 14 million residents.",
            source_ids=["S1"],
            origin="verification",
        )
    )

    class Client:
        def stream(self, question):
            assert question == EXAMPLES[0]
            yield Progress(node="search", round=1)
            yield outcome

    monkeypatch.setattr("groundmark.client.LocalClient", Client)
    app = AppTest.from_file(UI).run()
    assert not app.exception
    assert app.title[0].value == "Groundmark"
    assert [tab.label for tab in app.tabs] == ["Ask", "Search details"]
    app.button(key="example_0").click().run()
    assert not app.exception
    assert app.text_area[0].value == EXAMPLES[0]
    assert not [button for button in app.button if button.key == "example_0"]
    assert [item.value for item in app.subheader] == [
        "Answer",
        "Claims",
        "Conflicts",
        "Sources",
        "Searches",
        "Timing",
        "Usage",
    ]
    answer = app.markdown[0].value
    assert answer.startswith("The plant opened in 2024.\u00a0[1](https://example.org/")
    assert '"Example source")' in answer
    assert not app.info
    assert any(
        row.value == "**Not settled by the sources:**\n\n- Costs of \\$5 and \\$6 \\| tax."
        for row in app.markdown
    )
    labels = [item.label for item in app.expander]
    assert labels[0] == ":green-badge[Supported] The plant opened in 2024."
    assert labels[1] == ":primary-background[**1**] Example source · example.org"
    assert labels[2] == ":primary-background[**2**] Unread source · example.org"
    assert "Full trace (JSON)" not in labels
    captions = [row.value for row in app.caption]
    assert "1 read for the answer · 1 retrieved only" in captions
    assert "Read for the answer · retrieved " in " ".join(captions)
    assert any(row.startswith("Retrieved, not read for the answer") for row in captions)
    assert "Noted by the model while writing the answer, not checked" in captions
    assert "Found by checking claims against sources" in captions
    assert any(row.value.startswith("Tokyo has 14 million residents. [1](") for row in app.markdown)
    assert any(
        row.value.startswith("**Opening date** · Source 1 and source 2 give different dates. [1](")
        for row in app.markdown
    )
    assert any(row.startswith("1 of 1 claim supported · ") for row in captions)
    assert not any("Tavily" in row or "$" in row for row in captions)
    assert not any("S1" in row.value or "S2" in row.value for row in app.markdown)
    assert any(row.value == "> opened in 2024" for row in app.markdown)
    assert not app.json
    assert any(
        row.value == "1 search round · stopped because the sources covered the question"
        for row in app.markdown
    )
    assert any(
        row.value.startswith("**Round 1** ·") and row.value.endswith("· 1 new source")
        for row in app.markdown
    )
    assert not any(row.value.startswith("Run ") for row in app.caption)


def test_unanswerable_shows_one_message_and_reason(outcome):
    outcome.summary = ""
    outcome.claims = []
    outcome.insufficient_evidence = True
    outcome.missing_information = ["The 2030 prize has not been awarded yet."]
    outcome.trace.stop_reason = "unanswerable"
    outcome.trace.synthesis_source_ids = []
    app = AppTest.from_file(UI)
    app.session_state["result"] = outcome
    app.run()
    assert not app.exception
    assert [item.value for item in app.info] == ["This can't be known yet as of today's date."]
    assert "The 2030 prize has not been awarded yet." in [row.value for row in app.caption]
    assert "Claims" not in [item.value for item in app.subheader]
    assert app.subheader[0].value == "Answer"


def test_escape_blocks_markdown_structure():
    from groundmark.ui import _escape

    assert _escape("Title\n===\n- item\n+ more!") == "Title\n\\=\\=\\=\n\\- item\n\\+ more\\!"


def test_empty_question_does_not_start(monkeypatch):
    class Client:
        def stream(self, question):
            raise AssertionError("research should not start")

    monkeypatch.setattr("groundmark.client.LocalClient", Client)
    app = AppTest.from_file(UI).run()
    app.button[0].click().run()
    assert [warning.value for warning in app.warning] == ["Enter a question first."]
    assert app.text_area[0].proto.max_chars == 500
    assert not app.error
    assert not app.exception


def test_sources_are_numbered_in_answer_order(outcome):
    from groundmark.ui import _numbers, _plain

    first = outcome.sources[0]
    outcome.sources = [
        first.model_copy(update={"source_id": "S1"}),
        first.model_copy(update={"source_id": "S2"}),
        first.model_copy(update={"source_id": "S3"}),
        first.model_copy(update={"source_id": "S4"}),
    ]
    outcome.summary = "First. [S3]\n\nSecond. [S2] [S3]"
    outcome.trace.synthesis_source_ids = ["S4", "S3", "S2"]
    numbers = _numbers(outcome)
    assert numbers == {"S3": 1, "S2": 2, "S4": 3, "S1": 4}
    assert _plain("S4 and [S1] differ; S9 and SS2 are unknown. S3 too.", numbers) == (
        "Source 3 and source 4 differ; S9 and SS2 are unknown. Source 1 too."
    )


def test_favicon_ships_with_the_package():
    from groundmark.ui import FAVICON

    assert FAVICON.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_lowered_verdict_hides_quote_and_dates_are_readable(outcome):
    pair = outcome.claims[0].pairs[0]
    outcome.claims[0].pairs[0] = pair.model_copy(
        update={
            "verdict": "not_supported",
            "raw_verdict": "supported",
            "quote_match": "missing",
            "evidence_quote": "made up ... quote",
        }
    )
    outcome.sources[0].published_date = "Thu, 25 Jun 2026 00:00:00 GMT"
    outcome.sources[0].retrieved_at = "2026-11-20T20:48:54.506036+00:00"
    app = AppTest.from_file(UI)
    app.session_state["result"] = outcome
    app.run()
    assert not app.exception
    assert not any(row.value.startswith("> ") for row in app.markdown)
    captions = [row.value for row in app.caption]
    assert "Unverified quote: “made up ... quote”" in captions
    assert any("said this source supports the claim" in row for row in captions)
    assert any(
        "published Jun 25, 2026 · retrieved Nov 20, 2026, 20\\:48 UTC" in row for row in captions
    )
