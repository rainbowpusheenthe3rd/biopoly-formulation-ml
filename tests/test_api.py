from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from biopoly.api import main as api_main


@pytest.fixture(scope="module")
def client(fast_model):
    # inject the fast model instead of loading a champion from disk
    api_main.STATE["model"] = fast_model
    api_main.STATE["version"] = 0
    return TestClient(api_main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_ok_and_renormalises(client):
    r = client.post(
        "/predict",
        json={
            "frac": {"PLA": 0.8, "PBS": 0.2},
            "additives": {"plasticizer": 0.05},
            "process_temp_c": 195,
            "process_time_min": 20,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body["predictions"]) and body["warnings"]  # renormalisation warning


def test_predict_rejects_unknown_polymer(client):
    r = client.post(
        "/predict",
        json={"frac": {"XYZ": 1.0}, "process_temp_c": 195, "process_time_min": 20},
    )
    assert r.status_code == 422


def test_design_returns_candidates(client):
    r = client.post(
        "/design",
        json={
            "target": {"tensile_strength_mpa": 45},
            "method": "bayesopt",
            "top_k": 2,
            "budget": 120,
        },
    )
    assert r.status_code == 200
    assert len(r.json()["candidates"]) == 2


def test_design_rejects_unknown_target(client):
    r = client.post("/design", json={"target": {"nonsense": 1.0}})
    assert r.status_code == 422
