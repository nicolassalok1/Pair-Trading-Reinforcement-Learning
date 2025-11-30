"""Logs display page."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from streamlit_app.utils.log_handler import clear_log, get_log  # noqa: E402


def _maybe_autorefresh(key: str, enabled: bool, interval_sec: int) -> None:
    """Trigger rerun when auto-refresh is enabled."""
    if not enabled:
        return
    now = time.time()
    last = st.session_state.get(key, 0)
    if now - last >= interval_sec:
        st.session_state[key] = now
        st.rerun()


def _render_lines(lines: List[str]) -> str:
    """Format log lines with basic coloring."""
    if not lines:
        return "<div class='log-box'>No logs yet.</div>"
    out = ["<div class='log-box'>"]
    for line in lines:
        text = line.rstrip()
        if "ERROR" in text.upper() or "EXCEPTION" in text.upper():
            out.append(f"<div class='log-error'>{text}</div>")
        else:
            out.append(f"<div>{text}</div>")
    out.append("</div>")
    return "\n".join(out)


def render_module_logs(module_name: str, auto_refresh: bool, refresh_sec: int, tail: int) -> None:
    """Render a single module log panel."""
    _maybe_autorefresh(f"{module_name}_auto", auto_refresh, refresh_sec)
    lines = get_log(module_name, tail=tail)
    st.markdown(_render_lines(lines), unsafe_allow_html=True)
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button(f"Clear {module_name} log", key=f"clear_{module_name}"):
            clear_log(module_name)
            st.success("Log cleared.")
            st.rerun()
    with col2:
        st.caption(f"Showing last {tail} lines from {module_name}.log")


def render() -> None:
    """Render the logs tab with module selectors."""
    st.subheader("Execution logs")
    with st.expander("Guide des champs (Logs)", expanded=False):
        st.markdown(
            "- **Tail lines** : nombre de lignes de log affichees pour chaque module.\n"
            "- **Auto-refresh** : si coche, la page se relance toutes les X secondes.\n"
            "- **Refresh every (sec)** : cadence de rafraichissement automatique.\n"
            "- **Clear <module> log** : efface le fichier de log du module choisi."
        )
    st.caption("Utilisez ce panneau pour surveiller NLP/Heston/RL. En cas d'erreur, les lignes en rouge sont mises en avant.")
    tail = st.slider("Tail lines", 50, 2000, 400, step=50)
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_sec = st.slider("Refresh every (sec)", 2, 15, 4)

    tabs = st.tabs(["Logs NLP", "Logs Heston", "Logs RL"])
    with tabs[0]:
        render_module_logs("nlp", auto_refresh, refresh_sec, tail)
    with tabs[1]:
        render_module_logs("heston", auto_refresh, refresh_sec, tail)
    with tabs[2]:
        render_module_logs("rl", auto_refresh, refresh_sec, tail)


if __name__ == "__main__":
    render()
