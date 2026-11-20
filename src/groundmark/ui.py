import json
import re
from urllib.parse import quote

import streamlit as st

from groundmark.client import HttpClient, LocalClient, ResearchClient, RunError
from groundmark.config import Settings
from groundmark.models import Progress, ResearchResult, Source
from groundmark.research import EXAMPLES
from groundmark.sources import canonical_url

TITLE = "Groundmark"
SUBTITLE = "Live-web research agent with claim-level source verification"
PLACEHOLDER = "Ask a question to research on the live web..."
LENGTH = "Questions are limited to 500 characters."
INSUFFICIENT = "Insufficient evidence in the retrieved sources to fully answer this question."
NO_SOURCES = "No relevant sources were returned for this query."
BUDGET = "Stopped early because the configured run budget was reached."
TRUNCATED = "Model output hit the token limit before completing structured output."
FAILED = "A provider request failed. Check the research trace for details."
BUSY = "Another research run is in progress. Try again shortly."

PROGRESS = {
    "plan": "Planning search queries...",
    "search": "Searching the web...",
    "assess": "Checking evidence coverage...",
    "extract": "Reading full pages for unverified claims...",
    "synthesize": "Drafting grounded answer...",
    "verify": "Verifying claims against sources...",
    "finalize": "Finalizing trace...",
}
STATUS = {
    "supported": "Supported",
    "conflicting": "Sources disagree",
    "contradicted": "Contradicted",
    "not_supported": "Not supported by retrieved text",
}
QUOTES = {
    "exact": "Exact quote verified",
    "normalized": "Quote verified (whitespace normalized)",
    "missing": "Quote not found in source text",
    "not_applicable": "No quote expected",
}
SCOPE = {"snippet": "Checked against search snippet", "page": "Checked against extracted page"}
CITATION = re.compile(r"\[(S[0-9]+)\]")


def _escape(value: str) -> str:
    value = re.sub(r"([\\\[\]*_()~:#<>])", r"\\\1", value)
    return value.replace(chr(96), "\\" + chr(96))


def _link(label: str, url: str) -> str:
    label = _escape(label)
    if not canonical_url(url):
        return label
    destination = quote(url, safe=":/?&=%#;+@,[]~!$'")
    return f"[{label}]({destination})"


def _cited(text: str, sources: dict[str, Source]) -> str:
    fragments = []
    at = 0
    for match in CITATION.finditer(text):
        fragments.append(_escape(text[at : match.start()]))
        source = sources.get(match[1])
        fragments.append(_link(match[0], source.url) if source else _escape(match[0]))
        at = match.end()
    fragments.append(_escape(text[at:]))
    return "".join(fragments)


def _render_result(result: ResearchResult) -> None:
    sources = {source.source_id: source for source in result.sources}
    trace = result.trace
    kinds = {error.kind for error in trace.errors}
    st.subheader("Answer")
    if result.summary:
        st.markdown(_cited(result.summary, sources))
    if result.insufficient_evidence and result.sources:
        st.info(INSUFFICIENT)
        for item in result.missing_information:
            st.markdown(_escape(item))
    if not result.sources and trace.stop_reason == "no_sources":
        st.info(NO_SOURCES)
    if trace.stop_reason == "budget":
        st.warning(BUDGET)
    if "truncation" in kinds:
        st.warning(TRUNCATED)
    if kinds - {"budget", "truncation"}:
        st.warning(FAILED)

    st.subheader("Claim verification")
    for row in result.claims:
        with st.container(border=True):
            st.markdown(f"**{STATUS[row.status]}**")
            st.markdown(_escape(row.claim.text))
            if row.reason:
                st.text(row.reason)
            for pair in row.pairs:
                source = sources[pair.source_id]
                st.markdown(f"{_link(pair.source_id, source.url)} · **{STATUS[pair.verdict]}**")
                if pair.evidence_quote:
                    st.markdown(f"> {_escape(pair.evidence_quote)}")
                citation_note = "" if pair.cited else " · not cited by the answer"
                st.caption(
                    f"{QUOTES[pair.quote_match]} · {SCOPE[pair.evidence_scope]}{citation_note}"
                )
                st.markdown(_escape(pair.reason))

    if result.conflicts:
        st.subheader("Conflicts detected")
        for conflict in result.conflicts:
            st.markdown(f"**{_escape(conflict.topic)}** · {_escape(conflict.description)}")
            st.markdown(" · ".join(_link(ref, sources[ref].url) for ref in conflict.source_ids))

    with st.expander("Sources"):
        for source in result.sources:
            st.markdown(_link(f"[{source.source_id}] {source.title}", source.url))
            dates = " · ".join(
                value for value in (source.published_date, source.retrieved_at) if value
            )
            st.text(dates)
            st.markdown(_escape(source.content[:650]))

    with st.expander("Research trace"):
        st.json(
            {
                "run_id": trace.run_id,
                "model": trace.model,
                "plan_rationale": trace.plan_rationale,
                "rounds": [row.model_dump() for row in trace.rounds],
                "assessments": [row.model_dump() for row in trace.assessments],
                "stop_reason": trace.stop_reason,
                "stage_latency_ms": trace.stage_latency_ms,
                "total_latency_ms": trace.total_latency_ms,
                "tokens": trace.tokens.model_dump(),
                "tavily_credits": trace.tavily_credits,
                "estimated_usd": trace.estimated_usd,
                "claim_status_counts": trace.claim_status_counts,
                "quote_match_counts": trace.quote_match_counts,
                "errors": [error.model_dump() for error in trace.errors],
            }
        )
        st.dataframe(
            [
                {
                    "name": span.name,
                    "duration_ms": round(span.duration_ms, 2),
                    "attributes": json.dumps(span.attributes, ensure_ascii=False),
                }
                for span in trace.spans
            ],
            hide_index=True,
        )
        st.download_button(
            "Download redacted trace (JSON)",
            trace.model_dump_json(indent=2),
            file_name=f"groundmark-{trace.run_id}.json",
            mime="application/json",
        )


def _fill(question: str) -> None:
    st.session_state.question = question


def _show_progress(client: ResearchClient, question: str):
    slot = st.empty()
    slot.info(PROGRESS["plan"])
    for event in client.stream(question):
        if isinstance(event, ResearchResult):
            st.session_state.result = event
        elif isinstance(event, RunError):
            st.session_state.failure = event.kind
        elif isinstance(event, Progress) and event.node in PROGRESS:
            slot.info(
                "Running a follow-up search..."
                if event.node == "search" and event.round > 1
                else PROGRESS[event.node]
            )
    slot.empty()


def main(client: ResearchClient | None = None) -> None:
    st.set_page_config(page_title=TITLE, layout="centered", initial_sidebar_state="collapsed")
    st.markdown(
        "<style>.stMainBlockContainer{max-width:780px;}</style>",
        unsafe_allow_html=True,
    )
    st.title(TITLE)
    st.caption(SUBTITLE)
    question = st.text_input(
        PLACEHOLDER, placeholder=PLACEHOLDER, key="question", label_visibility="collapsed"
    )
    research = st.button("Research", type="primary", disabled=not question.strip())
    for example in EXAMPLES:
        st.button(example, on_click=_fill, args=(example,))

    if research:
        st.session_state.pop("result", None)
        st.session_state.pop("failure", None)
        if len(question) > 500:
            st.session_state.failure = "invalid_question"
        else:
            if client is None:
                api_url = Settings().api_url
                client = HttpClient(api_url) if api_url else LocalClient()
            _show_progress(client, question)
    if failure := st.session_state.get("failure"):
        st.error({"invalid_question": LENGTH, "busy": BUSY}.get(failure, FAILED))
    if result := st.session_state.get("result"):
        _render_result(result)


if __name__ == "__main__":
    main()
