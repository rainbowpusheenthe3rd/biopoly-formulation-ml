from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from biopoly.api import main as api_main
from biopoly.api import tenancy

pytestmark = pytest.mark.layer(9)  # inverse design & API

ACME = {"X-API-Key": "demo-acme-key"}  # scientist, tenant "acme"
GLOBEX = {"X-API-Key": "demo-globex-key"}  # scientist, tenant "globex"
ADMIN = {"X-API-Key": "demo-admin-key"}  # admin, tenant "acme"


@pytest.fixture(scope="module")
def client(fast_model):
    # inject the fast model instead of loading a champion from disk
    api_main.STATE["model"] = fast_model
    api_main.STATE["version"] = 0
    return TestClient(api_main.app)


@pytest.fixture(autouse=True)
def _fresh_tenancy():
    # each test starts with a clean tenant store + empty run log
    tenancy.reset_tenancy()
    yield
    tenancy.reset_tenancy()


def _predict_body() -> dict:
    return {
        "frac": {"PLA": 0.8, "PBS": 0.2},
        "additives": {"plasticizer": 0.05},
        "process_temp_c": 195,
        "process_time_min": 20,
    }


def test_health_is_open(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_requires_api_key(client):
    assert client.post("/predict", json=_predict_body()).status_code == 401


def test_predict_rejects_unknown_key(client):
    r = client.post("/predict", json=_predict_body(), headers={"X-API-Key": "nope"})
    assert r.status_code == 401


def test_whoami_resolves_tenant(client):
    r = client.get("/whoami", headers=ACME)
    assert r.status_code == 200
    body = r.json()
    assert body["tenant_id"] == "acme" and body["role"] == "scientist"


def test_predict_ok_and_renormalises(client):
    r = client.post("/predict", json=_predict_body(), headers=ACME)
    assert r.status_code == 200
    body = r.json()
    assert set(body["predictions"]) and body["warnings"]  # renormalisation warning


def test_predict_rejects_unknown_polymer(client):
    r = client.post(
        "/predict",
        json={"frac": {"XYZ": 1.0}, "process_temp_c": 195, "process_time_min": 20},
        headers=ACME,
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
        headers=ACME,
    )
    assert r.status_code == 200
    assert len(r.json()["candidates"]) == 2


def test_design_rejects_unknown_target(client):
    r = client.post("/design", json={"target": {"nonsense": 1.0}}, headers=ACME)
    assert r.status_code == 422


def test_history_is_tenant_isolated(client):
    # acme runs a prediction; globex must not see it, and vice versa
    assert client.post("/predict", json=_predict_body(), headers=ACME).status_code == 200
    acme_hist = client.get("/history", headers=ACME).json()
    globex_hist = client.get("/history", headers=GLOBEX).json()
    assert len(acme_hist["runs"]) == 1 and acme_hist["tenant_id"] == "acme"
    assert globex_hist["runs"] == [] and globex_hist["tenant_id"] == "globex"


def test_admin_usage_requires_admin_role(client):
    # a scientist key is forbidden; the admin key sees per-tenant counts
    assert client.get("/admin/usage", headers=ACME).status_code == 403
    client.post("/predict", json=_predict_body(), headers=ACME)
    usage = client.get("/admin/usage", headers=ADMIN).json()["usage"]
    assert usage.get("acme", 0) >= 1


def test_daily_quota_enforced(client, monkeypatch):
    monkeypatch.setattr(tenancy.settings, "tenant_daily_quota", 1)
    assert client.post("/predict", json=_predict_body(), headers=GLOBEX).status_code == 200
    # second call the same day is over quota
    assert client.post("/predict", json=_predict_body(), headers=GLOBEX).status_code == 429
