import re
from html import escape as html_escape
from urllib.parse import quote, urlsplit

import streamlit as st

from groundmark.client import HttpClient, LocalClient, ResearchClient, RunError
from groundmark.config import Settings
from groundmark.models import ClaimResult, Progress, ResearchResult, Source
from groundmark.research import EXAMPLES
from groundmark.sources import canonical_url

TITLE = "Groundmark"
SUBTITLE = (
    "Ask a question about anything on the web. "
    "Each claim in the answer is checked against its sources."
)
HEADER_HELP = (
    "Groundmark searches the live web with Tavily. Nemotron-3.5-Super-VL on Nebius Token "
    "Factory plans the searches, writes the answer and checks it. Answers are in English.\n\n"
    "The answer is split into short claims. Each claim is checked against every source read "
    "for the answer. A source supports or contradicts a claim only if the checker quotes it "
    "and the quote is found in the source text. Only supported claims appear in the answer.\n\n"
    "The same model writes and checks the claims, so a verdict is its judgment plus a quote "
    "match, not proof. Open the source before you rely on a figure."
)
PLACEHOLDER = "Ask a question about anything on the web"
EMPTY = "Enter a question first."
LENGTH = "Questions are limited to 500 characters."
BUSY = "Another question is being answered. Groundmark runs one at a time, so try again soon."
CONFIGURATION = (
    "Groundmark isn't configured. Add NEBIUS_API_KEY and TAVILY_API_KEY to .env and restart."
)
FAILED = "Something went wrong. Try again in a moment."
INSUFFICIENT = "The retrieved sources don't fully answer this question."
UNANSWERABLE = "This can't be known yet as of today's date."
NO_SOURCES = "Search returned no usable sources for this question."
BUDGET = "Stopped early because the run budget was reached."
TRUNCATED = "The model ran out of output tokens before finishing."
PROVIDER = "A provider request failed. The Search details tab has more."
GAPS = "Not found in the sources:"

PROGRESS = {
    "plan": "Planning searches…",
    "search": "Searching the web…",
    "assess": "Checking whether the sources answer the question…",
    "synthesize": "Writing claims from the sources…",
    "verify": "Checking each claim against the sources…",
    "extract": "Reading full pages for claims the snippets didn't support…",
    "finalize": "Finishing up…",
}
FOLLOW_UP = "Running a follow-up search…"
STATUS = {
    "supported": ("green", "Supported"),
    "conflicting": ("orange", "Sources disagree"),
    "contradicted": ("red", "Contradicted"),
    "not_supported": ("gray", "Not supported"),
}
VERDICT = {
    "supported": "Supports the claim",
    "contradicted": "Contradicts the claim",
    "not_supported": "Doesn't support the claim",
}
QUOTES = {
    "exact": "exact quote found in the source",
    "normalized": "quote found after normalizing spacing and punctuation",
    "missing": "quote not found in the source, so the verdict was lowered",
    "not_applicable": "",
}
SCOPE = {"snippet": "Checked against the search snippet", "page": "Checked against the full page"}
STOPS = {
    "sufficient": "the sources covered the question",
    "max_rounds": "it reached the two-round search limit",
    "no_new_query": "it had no new search to try",
    "unanswerable": "the answer can't be known yet",
    "no_sources": "search returned no usable sources",
    "budget": "the run budget was reached",
    "provider_error": "a provider request failed",
    "completed": "it finished",
}
STAGES = {
    "plan": "Plan",
    "search": "Search",
    "assess": "Assess",
    "synthesize": "Write claims",
    "verify": "Check claims",
    "extract": "Read full pages",
    "finalize": "Finalize",
}
CITATION = re.compile(r"([ \t]*)\[(S[0-9]+)\]")
MENTION = re.compile(r"\[?\b(S[0-9]+)\b\]?")

STYLES = """<style>
    [data-testid="stMainBlockContainer"]{max-width:768px;padding:3rem 24px 2rem}
    [data-testid="stCaptionContainer"],[data-testid="stCaptionContainer"] p,
    [role="tooltip"],[role="tooltip"] p{color:#1f1f1f!important;opacity:1}
    [data-testid="stCaptionContainer"] p{font-size:15px}
    .st-key-answer_stats [data-testid="stCaptionContainer"] p{font-size:13px}
    [data-testid="stExpanderDetails"] [data-testid="stCaptionContainer"] p{font-size:13px;
        color:#5f6368!important}
    .st-key-answer_text [data-testid="stMarkdownContainer"] :is(p,ul,ol,li){line-height:1.5}
    .st-key-answer_text [data-testid="stMarkdownContainer"] p{margin-bottom:.6rem}
    [data-testid="stTooltipIcon"],[data-testid="stTooltipIcon"] button{color:#5f6368;opacity:1}
    [role="tooltip"] p{margin-bottom:0}
    [role="tooltip"] p+p{margin-top:.5em}
    [data-testid="stTextAreaRootElement"]{background:#fff;border:1px solid #c7cace;
        border-radius:16px;box-shadow:0 2px 8px #1f1f1f0a}
    [data-testid="stTextArea"] textarea{font-size:17px;background:transparent;border:0;
        padding:12px 14px}
    [data-testid="stTextArea"] textarea::placeholder{color:#6b7177;opacity:1}
    :is(.st-key-answer_text,[class*="st-key-claim_"],.st-key-conflicts)
        [data-testid="stMarkdownContainer"] a{background:#fdecea;color:#a8190d;font-size:12px;
        font-weight:600;border-radius:6px;padding:1px 6px;text-decoration:none;
        white-space:nowrap}
    :is(.st-key-answer_text,[class*="st-key-claim_"],.st-key-conflicts)
        [data-testid="stMarkdownContainer"] a:hover{background:#fbd5d0}
    [class*="st-key-source_"] [data-testid="stMarkdownContainer"] a{color:#a8190d;
        text-decoration:none}
    [class*="st-key-source_"] [data-testid="stMarkdownContainer"] a:hover{text-decoration:underline}
    [data-testid="stExpander"] details{border:0;background:#f5f5f5;border-radius:12px}
    [data-testid="stExpander"] [data-testid="stText"]{font-family:inherit;font-size:14.5px;
        line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere}
    [data-testid="stExpander"] blockquote{border-left:3px solid #df2f1f;margin:.25rem 0;
        padding:.1rem .75rem}
    [data-testid="stExpander"] blockquote p{color:#1f1f1f}
    [data-testid="stExpanderDetails"] [data-testid="stVerticalBlock"]{gap:.45rem}
    [data-testid="stExpanderDetails"] hr{margin:.5rem 0}
    .st-key-examples .stButton button{background:#fff;color:#1f1f1f;border:1px solid #dfe1e4;
        border-radius:10px;text-align:left}
    .st-key-examples .stButton button:hover{border-color:#df2f1f;color:#a8190d}
    @keyframes groundmark-spin{to{transform:rotate(360deg)}}
    .groundmark-phase{display:flex;align-items:center;gap:8px;font-size:15px;color:#1f1f1f}
    .groundmark-phase::before{content:"";width:14px;height:14px;border:2px solid #df2f1f33;
        border-top-color:#df2f1f;border-radius:50%;animation:groundmark-spin .8s linear infinite}
    @media (max-width:640px){
        [data-testid="stMainBlockContainer"]{padding:1.5rem 16px 2rem}
        [data-testid="stTextArea"] textarea{font-size:16px}
    }
    </style>"""


def _escape(value: str) -> str:
    value = re.sub(r"([\\\[\]*_()~:#<>$|\-+=!{}])", r"\\\1", value)
    return value.replace(chr(96), "\\" + chr(96))


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _domain(url: str) -> str:
    host = urlsplit(url).hostname or ""
    return host.removeprefix("www.")


def _link(label: str, source: Source | None, url: str | None = None) -> str:
    label = _escape(label)
    url = url or (source.url if source else "")
    if not canonical_url(url):
        return label
    destination = quote(url, safe=":/?&=%#;+@,[]~!$'")
    if source is None:
        return f"[{label}]({destination})"
    title = re.sub(r"\s+", " ", source.title).replace("\\", "\\\\").replace('"', '\\"')
    return f'[{label}]({destination} "{title}")'


def _numbers(result: ResearchResult) -> dict[str, int]:
    """Number sources for display: cited in the answer, then read, then retrieved only."""
    known = [source.source_id for source in result.sources]
    read = set(result.trace.synthesis_source_ids)
    cited = [match[2] for match in CITATION.finditer(result.summary) if match[2] in known]
    order = [
        *dict.fromkeys(cited),
        *(source_id for source_id in known if source_id in read),
        *known,
    ]
    return {source_id: number for number, source_id in enumerate(dict.fromkeys(order), 1)}


def _plain(text: str, numbers: dict[str, int]) -> str:
    """Replace internal source IDs that the model wrote into free text."""
    return MENTION.sub(
        lambda match: f"source {numbers[match[1]]}" if match[1] in numbers else match[0], text
    )


def _chip(source_id: str, sources: dict[str, Source], numbers: dict[str, int]) -> str:
    if source_id not in numbers:
        return _escape(source_id)
    return _link(str(numbers[source_id]), sources[source_id])


def _cited(text: str, sources: dict[str, Source], numbers: dict[str, int]) -> str:
    fragments = []
    at = 0
    for match in CITATION.finditer(text):
        fragments.append(_escape(_plain(text[at : match.start()], numbers)))
        if match[2] in numbers:
            space = "\u00a0" if match[1] else ""
            fragments.append(space + _chip(match[2], sources, numbers))
        else:
            fragments.append(_escape(match[0]))
        at = match.end()
    fragments.append(_escape(_plain(text[at:], numbers)))
    return "".join(fragments)


def _phase(slot, text: str) -> None:
    slot.html(f'<div class="groundmark-phase" role="status">{html_escape(text)}</div>')


def _render_notices(result: ResearchResult, numbers: dict[str, int]) -> None:
    trace = result.trace
    kinds = {error.kind for error in trace.errors}
    if trace.stop_reason == "unanswerable":
        st.info(UNANSWERABLE)
        if result.missing_information:
            st.caption(_escape(_plain(result.missing_information[0], numbers)))
        return
    if not result.sources and trace.stop_reason == "no_sources":
        st.info(NO_SOURCES)
    elif result.insufficient_evidence and result.sources:
        st.info(INSUFFICIENT)
    if result.missing_information:
        bullets = "\n".join(
            f"- {_escape(_plain(item, numbers))}" for item in result.missing_information
        )
        st.markdown(f"**{GAPS}**\n\n{bullets}")
    if trace.stop_reason == "budget":
        st.warning(BUDGET)
    if "truncation" in kinds:
        st.warning(TRUNCATED)
    if kinds - {"budget", "truncation"}:
        st.warning(PROVIDER)


def _render_claim(
    index: int, row: ClaimResult, sources: dict[str, Source], numbers: dict[str, int]
) -> None:
    color, label = STATUS[row.status]
    title = f":{color}-badge[{label}] {_escape(row.claim.text)}"
    with st.expander(title, key=f"claim_{index}"):
        if row.reason:
            st.caption(_escape(_plain(row.reason, numbers)))
        if not row.pairs:
            st.caption("No source was checked for this claim.")
        pairs = sorted(row.pairs, key=lambda pair: numbers.get(pair.source_id, len(numbers) + 1))
        for number, pair in enumerate(pairs):
            if number:
                st.divider()
            chip = _chip(pair.source_id, sources, numbers)
            st.markdown(f"{chip} **{VERDICT[pair.verdict]}**")
            if pair.evidence_quote:
                st.markdown(f"> {_escape(pair.evidence_quote)}")
            st.markdown(_escape(_plain(pair.reason, numbers)))
            notes = [SCOPE[pair.evidence_scope]]
            if QUOTES[pair.quote_match]:
                notes.append(QUOTES[pair.quote_match])
            if not pair.cited:
                notes.append("found by cross-checking")
            st.caption(" · ".join(notes))


def _render_source(index: int, number: int, source: Source, read: bool) -> None:
    title = f":primary-background[**{number}**] {_escape(source.title)}"
    if domain := _domain(source.url):
        title += f" · {_escape(domain)}"
    with st.expander(title, key=f"source_{index}"):
        details = ["Read for the answer" if read else "Retrieved, not read for the answer"]
        if source.published_date:
            details.append(f"published {source.published_date}")
        details.append(f"retrieved {source.retrieved_at}")
        if source.evidence_scope == "page":
            details.append("full page fetched")
        st.caption(" · ".join(_escape(item) for item in details))
        text = source.content if len(source.content) <= 650 else source.content[:650] + "…"
        st.text(text)
        st.markdown(_link("Open page ↗", None, source.url))


def _render_result(result: ResearchResult) -> None:
    sources = {source.source_id: source for source in result.sources}
    numbers = _numbers(result)
    trace = result.trace
    if result.summary:
        with st.container(key="answer_text"):
            st.markdown(_cited(result.summary, sources, numbers))
    supported = sum(row.status == "supported" for row in result.claims)
    stats = []
    if result.claims:
        stats.append(f"{supported} of {_plural(len(result.claims), 'claim')} supported")
    stats.append(f"{trace.total_latency_ms / 1000:.1f} s")
    with st.container(key="answer_stats"):
        st.caption(" · ".join(stats))
    _render_notices(result, numbers)

    if result.claims:
        st.subheader(
            "Claims",
            help=(
                "Open a claim to see how each source was judged: the verdict, the quote the "
                "checker relied on and its reason. Not supported means the retrieved text "
                "doesn't establish the claim, not that the claim is false."
            ),
        )
        for index, row in enumerate(result.claims):
            _render_claim(index, row, sources, numbers)

    if result.conflicts:
        st.subheader(
            "Conflicts",
            help="Points where sources disagree. They are shown here and kept out of the answer.",
        )
        with st.container(key="conflicts"):
            for conflict in result.conflicts:
                st.caption(
                    "Found by checking claims against sources"
                    if conflict.origin == "verification"
                    else "Noted by the model while writing the answer, not checked"
                )
                topic = _plain(conflict.topic, numbers)
                description = _plain(conflict.description, numbers)
                if description.startswith(topic):
                    text = _escape(description)
                else:
                    text = f"**{_escape(topic)}** · {_escape(description)}"
                ordered = sorted(conflict.source_ids, key=lambda ref: numbers.get(ref, 0))
                refs = " ".join(_chip(ref, sources, numbers) for ref in ordered)
                st.markdown(f"{text} {refs}".strip())

    if result.sources:
        read_ids = set(trace.synthesis_source_ids)
        read_count = sum(source.source_id in read_ids for source in result.sources)
        st.subheader(
            "Sources",
            help=(
                "Every search result kept for this run. Claims were written and checked only "
                "against the sources read for the answer."
            ),
        )
        st.caption(
            f"{read_count} read for the answer · {len(result.sources) - read_count} retrieved only"
        )
        ordered = sorted(result.sources, key=lambda source: numbers[source.source_id])
        for index, source in enumerate(ordered):
            _render_source(index, numbers[source.source_id], source, source.source_id in read_ids)


def _render_trace(result: ResearchResult | None) -> None:
    if result is None:
        st.write("Ask a question first. This tab shows how its answer was found.")
        return
    trace = result.trace
    numbers = _numbers(result)
    st.write(
        f"{_plural(len(trace.rounds), 'search round')} · "
        f"stopped because {STOPS.get(trace.stop_reason, trace.stop_reason)}"
    )

    st.subheader(
        "Searches",
        help="The queries sent to Tavily and the agent's decision after each round.",
    )
    for number, search in enumerate(trace.rounds):
        queries = "; ".join(f"“{_escape(query)}”" for query in search.queries)
        added = _plural(len(search.source_ids), "new source")
        st.markdown(f"**Round {search.round_index}** · {queries} · {added}")
        if number < len(trace.assessments):
            assessment = trace.assessments[number]
            reason = _escape(_plain(assessment.stop_reason, numbers))
            st.caption(f"{assessment.decision.capitalize()}: {reason}")

    st.subheader("Timing", help="Wall-clock time of each stage in this run.")
    rows = [
        {"Stage": STAGES.get(name, name), "Time": f"{trace.stage_latency_ms[name] / 1000:.2f} s"}
        for name in [*STAGES, *sorted(set(trace.stage_latency_ms) - set(STAGES))]
        if name in trace.stage_latency_ms
    ]
    rows.append({"Stage": "Total", "Time": f"{trace.total_latency_ms / 1000:.2f} s"})
    st.dataframe(rows, hide_index=True, width="content")

    st.subheader(
        "Usage",
        help=(
            "Tavily credits and model tokens for this run. The cost is estimated from the "
            "token counts and list prices, not taken from an invoice."
        ),
    )
    tokens = trace.tokens
    st.dataframe(
        [
            {"Item": "Tavily credits", "Value": str(trace.tavily_credits)},
            {"Item": "Input tokens", "Value": f"{tokens.input:,}"},
            {"Item": "Cached input tokens", "Value": f"{tokens.cached_input:,}"},
            {"Item": "Output tokens (incl. reasoning)", "Value": f"{tokens.output:,}"},
            {"Item": "Reasoning tokens", "Value": f"{tokens.reasoning:,}"},
            {"Item": "Estimated model cost", "Value": f"${trace.estimated_usd:.4f}"},
        ],
        hide_index=True,
        width="content",
    )
    st.download_button(
        "Download trace (JSON)",
        trace.model_dump_json(indent=2),
        file_name=f"groundmark-{trace.run_id}.json",
        mime="application/json",
        type="tertiary",
        help=(
            "The full record of this run: run ID, queries, decisions, timings, tokens, errors "
            "and spans. API keys, hidden reasoning and page text are left out."
        ),
    )


def _select_example(question: str) -> None:
    st.session_state.question = question
    st.session_state.submit_example = True
    st.session_state.asked_once = True


def _begin() -> None:
    if (st.session_state.get("question") or "").strip():
        st.session_state.asked_once = True


def _run(client: ResearchClient, question: str) -> None:
    st.session_state.asked_once = True
    st.session_state.pop("result", None)
    st.session_state.pop("failure", None)
    slot = st.empty()
    _phase(slot, PROGRESS["plan"])
    try:
        for event in client.stream(question):
            if isinstance(event, ResearchResult):
                st.session_state.result = event
            elif isinstance(event, RunError):
                st.session_state.failure = event.kind
            elif isinstance(event, Progress) and event.node in PROGRESS:
                follow_up = event.node == "search" and event.round > 1
                _phase(slot, FOLLOW_UP if follow_up else PROGRESS[event.node])
    finally:
        slot.empty()


def _render_research(client: ResearchClient | None) -> None:
    with st.form("ask_form", border=False):
        question = st.text_area(
            "Question",
            placeholder=PLACEHOLDER,
            height=80,
            max_chars=500,
            key="question",
            label_visibility="collapsed",
        )
        with st.container(horizontal=True, horizontal_alignment="right"):
            submitted = st.form_submit_button("Ask", type="primary", on_click=_begin)

    if not st.session_state.get("asked_once"):
        with st.container(key="examples"):
            st.caption("Try an example:")
            for number, example in enumerate(EXAMPLES):
                st.button(
                    example,
                    key=f"example_{number}",
                    on_click=_select_example,
                    args=(example,),
                )

    submitted = submitted or st.session_state.pop("submit_example", False)
    question = question or ""
    if submitted and not question.strip():
        st.warning(EMPTY)
    elif submitted and len(question) > 500:
        st.session_state.pop("result", None)
        st.session_state.failure = "invalid_question"
    elif submitted:
        if client is None:
            api_url = Settings().api_url
            client = HttpClient(api_url) if api_url else LocalClient()
        _run(client, question.strip())

    if failure := st.session_state.get("failure"):
        messages = {"invalid_question": LENGTH, "busy": BUSY, "configuration": CONFIGURATION}
        st.error(messages.get(failure, FAILED))
    if result := st.session_state.get("result"):
        _render_result(result)


def main(client: ResearchClient | None = None) -> None:
    st.set_page_config(page_title=TITLE, layout="centered", initial_sidebar_state="collapsed")
    st.html(STYLES)
    st.title(TITLE)
    st.caption(SUBTITLE, help=HEADER_HELP)
    ask_tab, details_tab = st.tabs(["Ask", "Search details"])
    with ask_tab:
        _render_research(client)
    with details_tab:
        _render_trace(st.session_state.get("result"))


if __name__ == "__main__":
    main()
