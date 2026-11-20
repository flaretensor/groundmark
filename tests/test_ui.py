from pathlib import Path

from streamlit.testing.v1 import AppTest

from groundmark.models import ConflictResult, Progress
from groundmark.research import EXAMPLES


def test_screen_renders_claims_and_trace(monkeypatch, outcome):
    outcome.missing_information = ["Costs of $5 and $6 | tax."]
    outcome.sources.append(
        outcome.sources[0].model_copy(update={"source_id": "S2", "title": "Unread source"})
    )
    outcome.conflicts.append(
        ConflictResult(
            topic="Opening date",
            description="The published timelines disagree.",
            source_ids=["S1"],
            origin="synthesis",
        )
    )

    class Client:
        def stream(self, question):
            assert question == EXAMPLES[0]
            yield Progress(node="search", round=1)
            yield outcome

    monkeypatch.setattr("groundmark.client.LocalClient", Client)
    app = AppTest.from_file(Path("src/groundmark/ui.py").resolve()).run()
    assert not app.exception
    assert app.title[0].value == "Groundmark"
    app.button[1].click().run()
    assert app.text_input[0].value == EXAMPLES[0]
    app.button[0].click().run()
    assert not app.exception
    assert [item.value for item in app.subheader] == [
        "Answer",
        "Claim verification",
        "Conflicts detected",
    ]
    assert any("example.org" in row.value for row in app.markdown)
    assert not app.info
    assert any(
        row.value == "**Not found in retrieved sources:**\n\n- Costs of \\$5 and \\$6 \\| tax."
        for row in app.markdown
    )
    assert any(row.value.endswith(" · used in answer") for row in app.markdown)
    assert any(row.value.endswith(" · retrieved, not read for the answer") for row in app.markdown)
    assert any(
        row.value == "Reported by the model, not independently verified" for row in app.caption
    )
    assert [item.label for item in app.expander] == ["Sources", "Research trace"]
    assert "sufficient" in app.json[0].value


def test_unanswerable_message_accompanies_insufficient_evidence(outcome):
    outcome.summary = ""
    outcome.claims = []
    outcome.insufficient_evidence = True
    outcome.trace.stop_reason = "unanswerable"
    outcome.trace.synthesis_source_ids = []
    app = AppTest.from_file(Path("src/groundmark/ui.py").resolve())
    app.session_state["result"] = outcome
    app.run()
    assert not app.exception
    assert [item.value for item in app.info] == [
        "Insufficient evidence in the retrieved sources to fully answer this question.",
        "This can't be known yet as of today's date.",
    ]


def test_rejects_long_question_before_start():
    app = AppTest.from_file(Path("src/groundmark/ui.py").resolve()).run()
    app.text_input[0].set_value("a" * 501).run()
    app.button[0].click().run()
    assert [error.value for error in app.error] == ["Questions are limited to 500 characters."]
    assert not app.exception
