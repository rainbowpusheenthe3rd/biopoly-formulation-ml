"""Minimal multi-tenant login demo for the biopoly API (Streamlit).

A thin, login-gated UI over the FastAPI service, which stays the source of truth. It
demonstrates the multi-tenant UX sketched in ``docs/MULTI_TENANCY.md``: log in as a
tenant, submit a formulation for prediction or a target spec for inverse design, and
review *this tenant's* history.

Scope, honestly: tenant isolation here is at the **session layer** — a demo. Real
isolation (tenant_id on every row + Postgres row-level security + JWK/API-key auth) is
the documented next step, not built here.

Run (two terminals):
    uv run --extra frontend uvicorn biopoly.api.main:app          # 1: the API
    uv run --extra frontend streamlit run frontend/streamlit_app.py   # 2: this UI
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("BIOPOLY_API_URL", "http://localhost:8000")

# Demo tenants ONLY — not real authentication. A real deployment would issue per-tenant
# API keys + short-lived JWTs (see docs/MULTI_TENANCY.md), never hard-coded passwords.
_DEMO_TENANTS = {"acme": "demo", "globex": "demo"}

POLYMERS = ["PLA", "PHA", "PBAT", "PBS", "TPS", "PCL"]
TARGETS = [
    "tensile_strength_mpa",
    "melt_flow_index_g10min",
    "biodegradation_60d_pct",
    "water_absorption_pct",
    "optical_clarity_pct",
]


def _api_post(path: str, payload: dict) -> dict | None:
    """POST to the API, returning parsed JSON or None (with a UI error) on failure."""
    try:
        r = httpx.post(f"{API_URL}{path}", json=payload, timeout=30.0)
    except httpx.HTTPError as exc:
        st.error(f"Could not reach the API at {API_URL} ({exc}). Is it running?")
        return None
    if r.status_code >= 400:
        st.error(f"API {r.status_code}: {r.text}")
        return None
    return r.json()


def _login() -> None:
    st.title("biopoly — tenant login")
    st.caption("Demo login (try **acme / demo**). Real auth would use per-tenant API keys + JWTs.")
    tenant = st.text_input("Tenant", value="acme")
    password = st.text_input("Password", type="password")
    if st.button("Log in"):
        if _DEMO_TENANTS.get(tenant) == password:
            st.session_state["tenant"] = tenant
            st.session_state.setdefault("history", {})
            st.rerun()
        else:
            st.error("Invalid tenant/password.")


def _predict_tab(tenant: str) -> None:
    st.subheader("Predict properties")
    cols = st.columns(3)
    frac = {}
    for i, p in enumerate(POLYMERS):
        frac[p] = cols[i % 3].number_input(f"frac {p}", 0.0, 1.0, 0.5 if p == "PLA" else 0.0, 0.05)
    temp = st.slider("process temp (C)", 80, 260, 195)
    time_min = st.slider("process time (min)", 1, 120, 20)
    if st.button("Predict"):
        payload = {
            "frac": {p: v for p, v in frac.items() if v > 0},
            "process_temp_c": temp,
            "process_time_min": time_min,
        }
        out = _api_post("/predict", payload)
        if out:
            rows = [
                {"property": t, **{k: round(v, 2) for k, v in out["predictions"][t].items()}}
                for t in out["predictions"]
            ]
            st.table(rows)
            for w in out.get("warnings", []):
                st.info(w)
            st.session_state["history"].setdefault(tenant, []).append(("predict", payload))


def _design_tab(tenant: str) -> None:
    st.subheader("Inverse design (target -> formulation)")
    target = {}
    for t in TARGETS:
        if st.checkbox(t, value=(t == "tensile_strength_mpa")):
            target[t] = st.number_input(f"target {t}", value=45.0)
    if st.button("Design"):
        if not target:
            st.warning("Pick at least one target property.")
            return
        out = _api_post("/design", {"target": target, "method": "bayesopt", "top_k": 3})
        if out:
            for c in out["candidates"]:
                st.write(f"**score {c['score']}** — {c['formulation']}")
                st.caption(str(c["predicted"]))
            st.session_state["history"].setdefault(tenant, []).append(("design", target))


def _history_tab(tenant: str) -> None:
    st.subheader(f"History — tenant '{tenant}'")
    hist = st.session_state.get("history", {}).get(tenant, [])
    if not hist:
        st.caption("No runs yet this session.")
    for kind, payload in reversed(hist):
        st.write(f"- **{kind}** — {payload}")


def main() -> None:
    st.set_page_config(page_title="biopoly", layout="centered")
    if "tenant" not in st.session_state:
        _login()
        return
    tenant = st.session_state["tenant"]
    st.sidebar.write(f"Tenant: **{tenant}**")
    st.sidebar.caption(f"API: {API_URL}")
    if st.sidebar.button("Log out"):
        del st.session_state["tenant"]
        st.rerun()
    tab_predict, tab_design, tab_history = st.tabs(["Predict", "Design", "History"])
    with tab_predict:
        _predict_tab(tenant)
    with tab_design:
        _design_tab(tenant)
    with tab_history:
        _history_tab(tenant)


main()
