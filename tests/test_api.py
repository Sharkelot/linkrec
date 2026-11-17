"""API contract tests for the linkrec FastAPI service.

Run:  .venv/bin/python -m pytest tests/ -v
(imports serve.app, which loads all artifacts from outputs/ - a few seconds)
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from serve.app import app  # noqa: E402  (builds the app at import)

client = TestClient(app)
OUT = ROOT / "outputs"


@pytest.fixture(scope="module")
def split_pairs():
    return json.loads((OUT / "recsys_split_pairs.json").read_text())


def test_dashboard_200():
    r = client.get("/")
    assert r.status_code == 200
    assert "linkrec" in r.text


def test_recommend_known_user(split_pairs):
    r = client.get("/recommend", params={"user_id": 196, "k": 10})
    assert r.status_code == 200
    d = r.json()
    assert d["user_id"] == 196
    assert len(d["recommendations"]) == 10
    # serving mask: no recommendation may be in the user's train history
    hist = {i for u, i in split_pairs["train"] if u == 196}
    ids = [rec["item_id"] for rec in d["recommendations"]]
    assert len(set(ids)) == 10
    assert not (set(ids) & hist)
    # raw ML-100k ids with real titles
    titles = json.loads((OUT / "recsys_item_titles.json").read_text())
    for rec in d["recommendations"]:
        assert 1 <= rec["item_id"] <= 1682
        assert rec["title"] == titles.get(str(rec["item_id"]))
    # scores strictly descending
    scores = [rec["score"] for rec in d["recommendations"]]
    assert scores == sorted(scores, reverse=True)


def test_recommend_unknown_user_404():
    assert client.get("/recommend", params={"user_id": 99999, "k": 10}).status_code == 404


def test_linkpred_top_pairs_not_connected():
    r = client.get("/linkpred")
    assert r.status_code == 200
    d = r.json()
    pairs = d["top_unconnected_edges"]
    assert len(pairs) == 20
    data = json.loads((OUT / "linkpred_data.json").read_text()) if (OUT / "linkpred_data.json").exists() else None
    for p in pairs:
        assert p["src"] != p["dst"]
        assert p["score"] < 0.9995  # duplicate-feature pairs must be excluded


def test_linkpred_score_bounds():
    r = client.get("/linkpred/score", params={"src": 0, "dst": 1})
    assert r.status_code == 200
    d = r.json()
    assert -1.0 <= d["cosine_score"] <= 1.0
    assert isinstance(d["is_known_edge"], bool)


def test_linkpred_score_invalid_400():
    assert client.get("/linkpred/score", params={"src": 0, "dst": 0}).status_code == 400


def test_metrics_match_artifacts():
    m = client.get("/metrics").json()
    reported = json.loads((OUT / "recsys_metrics.json").read_text())
    assert m["recsys"]["test"] == reported["test"]
    assert m["recsys"]["validation"] == reported["valid"]
    lp = json.loads((OUT / "linkpred_metrics.json").read_text())
    assert m["linkpred"]["auc_uniform"] == lp["auc_uniform"]
    assert "latency_local_ms" in m and "uptime_s" in m


def test_b2b_endpoint():
    r = client.get("/b2b")
    assert r.status_code == 200
    d = r.json()
    assert d["n_companies"] == 3000
    assert d["n_latent_edges"] >= 10000
    assert len(d["top_pairs"]) == 20
    for p in d["top_pairs"]:
        for field in ("src", "dst", "score", "src_sector", "dst_sector",
                      "common_neighbors", "is_latent_ground_truth"):
            assert field in p


def test_explain_returns_facts_or_503():
    r = client.get("/explain", params={"graph": "b2b", "src": 0, "dst": 1})
    assert r.status_code in (200, 503)  # 503 only when the local LLM is down
    d = r.json()
    assert d["facts"] is not None
    assert "same_sector" in d["facts"] and "common_train_partners" in d["facts"]
    if r.status_code == 200:
        assert d["explanation"] and len(d["explanation"]) > 50
        assert d["source"] == "local-llm"
    else:
        assert d["explanation"] is None


def test_explain_cora_and_bad_inputs():
    r = client.get("/explain", params={"graph": "cora", "src": 1, "dst": 2})
    assert r.status_code in (200, 503)
    assert client.get("/explain", params={"graph": "nope", "src": 0, "dst": 1}).status_code == 400
    assert client.get("/explain", params={"graph": "b2b", "src": 0, "dst": 0}).status_code == 400
