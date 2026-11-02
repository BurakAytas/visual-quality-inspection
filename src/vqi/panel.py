"""Operator panel: inspect a part, work the review queue, read the daily report.

A thin Streamlit front end over the inspection API; it holds no model and no data of its
own, so the panel and the line always see the same decisions.

Run (API address from ``VQI_API_URL``, default ``http://localhost:8000``):
    streamlit run src/vqi/panel.py
"""

import base64
import os
from typing import Any

import httpx
import pandas as pd
import streamlit as st

DEFAULT_API_URL = "http://localhost:8000"
# Colours of the decision banner.
DECISION_COLOURS = {"OK": "#1b7f3b", "NOK": "#b3261e"}


def decision_banner(body: dict[str, Any]) -> str:
    """HTML banner with the decision, and the defect type of a NOK part."""
    text = body["decision"]
    if body.get("defect_type"):
        text += f" &middot; {body['defect_type']} ({body['defect_probability']:.0%})"
    colour = DECISION_COLOURS[body["decision"]]
    return (
        f'<div style="background:{colour};color:white;padding:0.6rem 1rem;'
        f'border-radius:0.4rem;font-size:1.6rem;font-weight:600">{text}</div>'
    )


def show_result(body: dict[str, Any], image: bytes) -> None:
    """Decision, numbers and the image next to its heatmap."""
    st.markdown(decision_banner(body), unsafe_allow_html=True)
    if body["needs_review"]:
        st.warning("Score close to the threshold: an operator should look at this part.")
    numbers, original, heatmap = st.columns([1, 2, 2])
    numbers.metric("Score", f"{body['score']:.2f}")
    numbers.metric("Threshold", f"{body['threshold']:.2f}")
    numbers.metric("Latency", f"{body['latency_ms']:.0f} ms")
    original.image(image, caption=body["file"], width="stretch")
    overlay = base64.b64decode(body["heatmap_png_base64"])
    heatmap.image(overlay, caption="Anomaly heatmap", width="stretch")
    st.caption(f"Model {body['model_version']}")


def answer_once(
    client: httpx.Client, answers: dict[str, Any], file_id: str, name: str, image: bytes
) -> dict[str, Any]:
    """The API's answer for an upload, posted only the first time ``file_id`` is seen.

    Streamlit reruns the script on every click; without this the part would be inspected
    and logged again each time. Raises ``httpx.HTTPStatusError`` if the API refuses it.
    """
    if file_id not in answers:
        response = client.post("/inspect", files={"file": (name, image)})
        response.raise_for_status()
        answers[file_id] = response.json()
    return answers[file_id]


def inspect_tab(client: httpx.Client) -> None:
    """Upload one image and show the API's answer."""
    upload = st.file_uploader("Part image", type=["png", "jpg", "jpeg", "bmp"])
    if upload is None:
        st.info("Upload an image of a part to inspect it.")
        return
    image = upload.getvalue()
    answers = st.session_state.setdefault("answers", {})
    try:
        body = answer_once(client, answers, upload.file_id, upload.name, image)
    except httpx.HTTPStatusError as error:
        st.error(error.response.json().get("detail", error.response.text))
        return
    show_result(body, image)


def review_tab(client: httpx.Client) -> None:
    """Parts whose score lies near the threshold, newest first."""
    rows = client.get("/review").json()
    if not rows:
        st.success("No parts waiting for review.")
        return
    columns = ["timestamp", "file", "decision", "score", "threshold", "defect_type"]
    st.dataframe(pd.DataFrame(rows)[columns], hide_index=True, width="stretch")


def report_tab(client: httpx.Client) -> None:
    """Counts and defect rate per day, and NOK parts per defect type."""
    daily = client.get("/daily").json()
    if not daily["days"]:
        st.info("No decisions logged yet.")
        return
    st.dataframe(pd.DataFrame(daily["days"]), hide_index=True, width="stretch")
    if daily["defect_types"]:
        types = pd.DataFrame(daily["defect_types"])
        st.bar_chart(types.pivot_table(index="defect_type", values="nok", aggfunc="sum"))


def render(client: httpx.Client) -> None:
    """The whole panel; the client is passed in so tests can fake the API."""
    st.title("Visual quality inspection")
    try:
        health = client.get("/health").json()
    except httpx.HTTPError as error:
        st.error(f"Inspection API not reachable: {error}")
        return
    st.caption(f"Serving {health['model_version']}")
    inspect, review, report = st.tabs(["Inspect", "Review queue", "Daily report"])
    with inspect:
        inspect_tab(client)
    with review:
        review_tab(client)
    with report:
        report_tab(client)


def main() -> None:
    """Entry point for ``streamlit run``."""
    st.set_page_config(page_title="Visual quality inspection", layout="wide")
    render(httpx.Client(base_url=os.environ.get("VQI_API_URL", DEFAULT_API_URL), timeout=30))


if __name__ == "__main__":
    main()
