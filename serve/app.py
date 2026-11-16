#!/usr/bin/env python3
"""FastAPI service for the linkrec project.

Serves:
  /               - dashboard (metrics, latency, top unconnected edges, rec form)
  /recommend      - GET ?user_id=N&k=10 -> top-k items (LightGCN embeddings)
  /linkpred       - GET -> top unconnected Cora edge candidates (GraphSAGE)
  /linkpred/score - GET ?src=A&dst=B -> predicted similarity for a pair
  /metrics        - GET -> eval metrics (NDCG@10, Recall@10, MRR, AUC) + latency

Runs:  .venv/bin/python -m uvicorn serve.app:app --host 127.0.0.1 --port 8620
"""
import json
import sys
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from torch_geometric.nn import SAGEConv

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- state
latency_log = deque(maxlen=4000)   # (path, seconds)
STARTED = time.time()


def _load_json(name, default):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else default


def build_app() -> FastAPI:
    app = FastAPI(title="linkrec", version="1.0")

    # ---- recsys (RecBole LightGCN on ML-100k, chronological split) ----
    emb = torch.load(OUT / "recsys_embeddings.pt", map_location="cpu", weights_only=True)
    user_emb = emb["user_emb"].to(DEVICE)
    item_emb = emb["item_emb"].to(DEVICE)
    known = torch.load(OUT / "recsys_known_train.pt", map_location="cpu", weights_only=True)
    known = known.to(DEVICE)
    titles = _load_json("recsys_item_titles.json", {})
    recsys_metrics = _load_json("recsys_metrics.json", {})
    num_items = item_emb.shape[0]

    # ---- linkpred (PyG GraphSAGE on Cora) ----
    lp = torch.load(OUT / "linkpred_model.pt", map_location="cpu", weights_only=False)
    lp_args = lp["args"]
    lp_data = torch.load(OUT / "linkpred_data.pt", map_location="cpu", weights_only=True)

    class SAGE(torch.nn.Module):
        def __init__(self, in_dim, hidden, out):
            super().__init__()
            self.conv1 = SAGEConv(in_dim, hidden)
            self.conv2 = SAGEConv(hidden, hidden)
            self.head = torch.nn.Linear(hidden, out)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index).relu()
            return self.head(x)

    lp_model = SAGE(lp_args["in_dim"], lp_args["hidden"], lp_args["out"]).to(DEVICE)
    lp_model.load_state_dict(lp["state_dict"])
    lp_model.eval()
    x = lp_data["x"].to(DEVICE)
    train_idx = lp_data["edge_index_train"].to(DEVICE)
    heldout_idx = lp_data["edge_index_heldout"].to(DEVICE)
    cora_names = lp_data["names"]
    with torch.no_grad():
        lp_emb = F.normalize(lp_model(x, train_idx), dim=-1)
    known_lp = torch.zeros(lp_data["num_nodes"], lp_data["num_nodes"], dtype=torch.bool, device=DEVICE)
    for idx in (train_idx, heldout_idx):
        known_lp[idx[0], idx[1]] = True
        known_lp[idx[1], idx[0]] = True
    known_lp.fill_diagonal_(True)

    linkpred_metrics = _load_json("linkpred_metrics.json", {})
    top_edges = _load_json("linkpred_top_edges.json", [])

    def record(path: str, dt: float):
        latency_log.append((path, dt))

    def latency_stats(prefix="/recommend"):
        vals = sorted(dt for p, dt in latency_log if p.startswith(prefix))
        if not vals:
            return {"n": 0}
        n = len(vals)
        q = lambda f: vals[min(n - 1, int(f * n))]
        return {
            "n": n,
            "p50_ms": round(q(0.50) * 1000, 2),
            "p90_ms": round(q(0.90) * 1000, 2),
            "p99_ms": round(q(0.99) * 1000, 2),
            "max_ms": round(vals[-1] * 1000, 2),
        }

    def latency_stats_all():
        vals = sorted(dt for _, dt in latency_log)
        if not vals:
            return {"n": 0}
        n = len(vals)
        q = lambda f: vals[min(n - 1, int(f * n))]
        return {
            "n": n,
            "p50_ms": round(q(0.50) * 1000, 2),
            "p90_ms": round(q(0.90) * 1000, 2),
            "p99_ms": round(q(0.99) * 1000, 2),
            "max_ms": round(vals[-1] * 1000, 2),
        }

    # ------------------------------------------------------------ routes
    @app.get("/recommend")
    def recommend(user_id: int = Query(...), k: int = Query(10, ge=1, le=100)):
        t0 = time.perf_counter()
        # RecBole kept raw 1-based ML-100k ids: users 1..943, items 1..1682
        if not (1 <= user_id < user_emb.shape[0]):
            record("/recommend", time.perf_counter() - t0)
            return JSONResponse(status_code=404, content={"error": f"user_id out of range 1..{user_emb.shape[0]-1}"})
        ui = user_id
        with torch.no_grad():
            scores = user_emb[ui] @ item_emb.T
            scores = scores.masked_fill(known[ui], float("-inf"))
            top = torch.topk(scores, min(k, int((~known[ui]).sum())))
        recs = []
        for item_idx, score in zip(top.indices.tolist(), top.values.tolist()):
            recs.append({
                "item_id": item_idx,
                "title": titles.get(str(item_idx), f"item {item_idx}"),
                "score": round(float(score), 4),
            })
        dt = time.perf_counter() - t0
        record("/recommend", dt)
        return {
            "user_id": user_id,
            "k": k,
            "recommendations": recs,
            "model": recsys_metrics.get("model", "LightGCN"),
            "latency_ms": round(dt * 1000, 2),
        }

    @app.get("/linkpred")
    def linkpred():
        t0 = time.perf_counter()
        with torch.no_grad():
            sim = lp_emb @ lp_emb.T
        sim = sim.masked_fill(known_lp, float("-inf"))
        tri = torch.triu_indices(lp_data["num_nodes"], lp_data["num_nodes"], offset=1, device=DEVICE)
        cand = sim[tri[0], tri[1]]
        vals, idx = torch.topk(cand, len(top_edges) or 20)
        out = [
            {"src": int(tri[0][idx[j]]), "dst": int(tri[1][idx[j]]), "score": round(float(vals[j]), 4),
             "src_title": cora_names[int(tri[0][idx[j]])], "dst_title": cora_names[int(tri[1][idx[j]])]}
            for j in range(len(idx))
        ]
        dt = time.perf_counter() - t0
        record("/linkpred", dt)
        return {"dataset": "Cora", "model": "GraphSAGE", "top_unconnected_edges": out,
                "latency_ms": round(dt * 1000, 2)}

    @app.get("/linkpred/score")
    def linkpred_score(src: int = Query(...), dst: int = Query(...)):
        t0 = time.perf_counter()
        if not (0 <= src < lp_data["num_nodes"] and 0 <= dst < lp_data["num_nodes"]) or src == dst:
            record("/linkpred", time.perf_counter() - t0)
            return JSONResponse(status_code=400, content={"error": "invalid src/dst"})
        with torch.no_grad():
            s = float((lp_emb[src] * lp_emb[dst]).sum())
        dt = time.perf_counter() - t0
        record("/linkpred", dt)
        return {"src": src, "dst": dst,
                "src_title": cora_names[src], "dst_title": cora_names[dst],
                "cosine_score": round(s, 4),
                "is_known_edge": bool(known_lp[src, dst]),
                "latency_ms": round(dt * 1000, 2)}

    @app.get("/metrics")
    def metrics():
        t0 = time.perf_counter()
        out = {
            "recsys": {
                "dataset": recsys_metrics.get("dataset"),
                "model": recsys_metrics.get("model"),
                "split": recsys_metrics.get("split"),
                "validation": recsys_metrics.get("valid"),
                "test": recsys_metrics.get("test"),
            },
            "linkpred": linkpred_metrics,
            "latency_local_ms": {
                "recommend": latency_stats("/recommend"),
                "linkpred": latency_stats("/linkpred"),
                "all": latency_stats_all(),
            },
            "uptime_s": round(time.time() - STARTED, 1),
        }
        dt = time.perf_counter() - t0
        record("/metrics", dt)
        return out

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return DASHBOARD_HTML

    return app


# ------------------------------------------------------------ dashboard
DASHBOARD_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>linkrec — embeddings &amp; link prediction</title>
<style>
:root{color-scheme:dark}
body{font:14px/1.5 system-ui,sans-serif;background:#0d1117;color:#e6edf3;margin:0;padding:24px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:15px;margin:28px 0 10px;color:#79c0ff}
.sub{color:#8b949e;margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}
.card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px}
.card .v{font-size:26px;font-weight:600}.card .l{color:#8b949e;font-size:12px;margin-top:2px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border:1px solid #30363d;padding:6px 9px;text-align:left}
th{background:#161b22}
input{background:#0d1117;border:1px solid #30363d;color:#e6edf3;border-radius:6px;padding:7px 10px}
button{background:#238636;color:#fff;border:0;border-radius:6px;padding:8px 14px;cursor:pointer}
.mono{font-family:ui-monospace,monospace}
.dim{color:#8b949e}
</style></head><body>
<h1>linkrec — embeddings &amp; link prediction</h1>
<div class="sub">PyG GraphSAGE link prediction (Cora) · RecBole LightGCN recommendations (ML-100k, chronological TIME split) · served behind Cloudflare</div>

<h2>Recommendation ranking quality (chronological test split)</h2>
<div class="grid" id="recsys-metrics"></div>

<h2>Link prediction (held-out edges, 20% never trained on)</h2>
<div class="grid" id="lp-metrics"></div>

<h2>Inference latency (local, measured per request)</h2>
<table id="latency"><tr><th>endpoint</th><th>n</th><th>p50</th><th>p90</th><th>p99</th><th>max</th></tr></table>

<h2>Top unconnected pairs the model thinks are related (Cora)</h2>
<table id="top-edges"><tr><th>score</th><th>paper A</th><th>paper B</th></tr></table>

<h2>Try a recommendation</h2>
<div style="display:flex;gap:8px;align-items:center">
<input id="uid" type="number" value="196" min="1" style="width:110px">
<input id="k2" type="number" value="10" min="1" max="100" style="width:70px">
<button onclick="doRec()">Recommend top-k</button>
<span class="dim" id="rec-lat"></span>
</div>
<div style="height:14px"></div>
<table id="recs"><tr><th>#</th><th>item</th><th>title</th><th>score</th></tr></table>

<script>
async function j(u){const r=await fetch(u);return r.json()}
function card(l,v){return `<div class="card"><div class="v">${v}</div><div class="l">${l}</div></div>`}
async function refresh(){
  const m=await j('/metrics');
  const t=m.recsys.test||{}, v=m.recsys.validation||{};
  document.getElementById('recsys-metrics').innerHTML=
    card('NDCG@10 (test)',t['NDCG@10']??'—')+card('Recall@10 (test)',t['Recall@10']??'—')+
    card('MRR@10 (test)',t['MRR@10']??'—')+card('NDCG@10 (val)',v['NDCG@10']??'—')+
    `<div class="card"><div class="v dim" style="font-size:13px;padding-top:8px">${m.recsys.dataset} · ${m.recsys.split||''}</div></div>`;
  const L=m.linkpred||{};
  document.getElementById('lp-metrics').innerHTML=
    card('AUC (held-out)',L.auc??'—')+card('MRR (held-out)',L.mrr??'—')+
    `<div class="card"><div class="v" style="font-size:16px;padding-top:10px">${L.heldout_edges??'—'} held-out edges</div><div class="l">${L.dataset} · ${L.model||''}</div></div>`;
  const lat=m.latency_local_ms||{};
  let rows='<tr><th>endpoint</th><th>n</th><th>p50</th><th>p90</th><th>p99</th><th>max</th></tr>';
  for(const [k,s] of Object.entries(lat)){
    if(!s.n) continue;
    rows+=`<tr><td class="mono">/${k==='all'?'*':k.replace('/','')}</td><td>${s.n}</td><td>${s.p50_ms} ms</td><td>${s.p90_ms} ms</td><td>${s.p99_ms} ms</td><td>${s.max_ms} ms</td></tr>`;
  }
  document.getElementById('latency').innerHTML=rows;
  const lp=await j('/linkpred');
  let er='';
  for(const e of lp.top_unconnected_edges){
    er+=`<tr><td class="mono">${e.score}</td><td>${e.src_title}</td><td>${e.dst_title}</td></tr>`;
  }
  document.getElementById('top-edges').innerHTML='<tr><th>score</th><th>paper A</th><th>paper B</th></tr>'+er;
}
async function doRec(){
  const u=document.getElementById('uid').value, k=document.getElementById('k2').value;
  const r=await j(`/recommend?user_id=${u}&k=${k}`);
  document.getElementById('rec-lat').textContent=`local ${r.latency_ms} ms`;
  let rows='';
  r.recommendations.forEach((x,i)=>rows+=`<tr><td>${i+1}</td><td class="mono">${x.item_id}</td><td>${x.title}</td><td class="mono">${x.score}</td></tr>`);
  document.getElementById('recs').innerHTML='<tr><th>#</th><th>item</th><th>title</th><th>score</th></tr>'+rows;
}
refresh();setInterval(refresh,5000);doRec();
</script>
</body></html>"""


app = build_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8620, log_level="info")
