from pathlib import Path

from streamlit.testing.v1 import AppTest

from groundmark.models import Progress
from groundmark.research import EXAMPLES


def test_screen_renders_claims_and_trace(monkeypatch, outcome):
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
    assert [item.value for item in app.subheader] == ["Answer", "Claim verification"]
    assert any("example.org" in row.value for row in app.markdown)
    assert [item.label for item in app.expander] == ["Sources", "Research trace"]
    assert "sufficient" in app.json[0].value


def test_rejects_long_question_before_start():
    app = AppTest.from_file(Path("src/groundmark/ui.py").resolve()).run()
    app.text_input[0].set_value("a" * 501).run()
    app.button[0].click().run()
    assert [error.value for error in app.error] == ["Questions are limited to 500 characters."]
    assert not app.exception
