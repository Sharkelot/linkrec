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
import os
import sys
import time
import urllib.request
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
    id_map = _load_json("recsys_id_map.json", {})
    # embeddings/known-mask live in RecBole's REMAPPED id space; the API
    # speaks RAW ML-100k ids. Convert at this boundary.
    user_r2r = {int(k): v for k, v in id_map.get("user_raw_to_remapped", {}).items()}
    item_r2r = {int(k): v for k, v in id_map.get("item_raw_to_remapped", {}).items()}
    item_r2raw = {v: k for k, v in item_r2r.items()}
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
    # identical raw-feature rows (bag-of-words duplicates) score ~cosine 1.0
    # without any graph signal - exclude them from candidate ranking
    _, _inv = torch.unique(x, dim=0, return_inverse=True)
    dup_lp = _inv.unsqueeze(0) == _inv.unsqueeze(1)

    linkpred_metrics = _load_json("linkpred_metrics.json", {})
    top_edges = _load_json("linkpred_top_edges.json", [])

    # ---- b2b (synthetic ground-truth graph) ----
    b2b_top = _load_json("b2b_top_pairs.json", [])
    b2b_metrics = _load_json("b2b_metrics.json", {})
    b2b_meta = _load_json("b2b_meta.json", {})
    b2b_graph = None
    if (OUT / "b2b_graph.pt").exists():
        b2b_graph = torch.load(OUT / "b2b_graph.pt", map_location="cpu")
    # local LLM for /explain: Ollama gemma3:4b by default (dedicated instance;
    # the 18081 Chon-12B llama-server is shared with the JEV decision flow).
    # Override with LINKREC_LLM_URL / LINKREC_LLM_MODEL env vars.
    LLM_URL = os.environ.get("LINKREC_LLM_URL",
                             "http://127.0.0.1:11434/v1/chat/completions")
    LLM_MODEL = os.environ.get("LINKREC_LLM_MODEL", "gemma3:4b")

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
        # API user ids are RAW ML-100k ids (1..943); embeddings are in
        # RecBole's remapped space - convert before indexing.
        ui = user_r2r.get(user_id)
        if ui is None:
            record("/recommend", time.perf_counter() - t0)
            return JSONResponse(status_code=404, content={"error": f"unknown raw user_id {user_id} (expected 1..943)"})
        with torch.no_grad():
            scores = user_emb[ui] @ item_emb.T
            scores = scores.masked_fill(known[ui], float("-inf"))
            top = torch.topk(scores, min(k, int((~known[ui]).sum())))
        recs = []
        for item_idx, score in zip(top.indices.tolist(), top.values.tolist()):
            raw_item = item_r2raw[item_idx]
            recs.append({
                "item_id": raw_item,
                "title": titles.get(str(raw_item), f"item {raw_item}"),
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
        sim = sim.masked_fill(known_lp | dup_lp, float("-inf"))
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

    @app.get("/b2b")
    def b2b():
        """Top unconnected B2B company pairs (pair_ranker scores) with
        relationship metadata + ground-truth annotation (demo benchmark)."""
        t0 = time.perf_counter()
        models = b2b_metrics.get("models", {})
        out = {
            "dataset": "synthetic B2B (ground-truth latent edges)",
            "n_companies": b2b_meta.get("n_companies"),
            "n_latent_edges": b2b_meta.get("n_latent_edges"),
            "model_headline": {m: r.get("precision@10") for m, r in models.items()},
            "top_pairs": b2b_top,
        }
        dt = time.perf_counter() - t0
        record("/b2b", dt)
        out["latency_ms"] = round(dt * 1000, 2)
        return out

    @app.get("/explain")
    def explain(src: int = Query(...), dst: int = Query(...),
                graph: str = Query("b2b")):
        """Local-LLM explanation of why the model thinks src~dst are related.

        facts = model-derived evidence (no LLM needed); the LLM only turns
        them into prose. Returns 503 with the facts if the LLM is down.
        """
        t0 = time.perf_counter()
        facts = None
        try:
            if graph == "b2b" and b2b_graph is not None:
                N = b2b_graph["x"].shape[0]
                if not (0 <= src < N and 0 <= dst < N) or src == dst:
                    return JSONResponse(status_code=400,
                                        content={"error": f"invalid pair (0..{N - 1})"})
                sec, reg, size = b2b_graph["sector"], b2b_graph["region"], b2b_graph["size"]
                adj = b2b_graph["train_adj"]
                cn = int((adj[src].int() * adj[dst].int()).sum())
                facts = {
                    "src_sector": b2b_meta["sector_names"][int(sec[src])],
                    "dst_sector": b2b_meta["sector_names"][int(sec[dst])],
                    "same_sector": bool(sec[src] == sec[dst]),
                    "src_region": b2b_meta["region_names"][int(reg[src])],
                    "dst_region": b2b_meta["region_names"][int(reg[dst])],
                    "same_region": bool(reg[src] == reg[dst]),
                    "src_size_log10": round(float(size[src]), 2),
                    "dst_size_log10": round(float(size[dst]), 2),
                    "common_train_partners": cn,
                    "deg_src": int(adj[src].sum()), "deg_dst": int(adj[dst].sum()),
                    "already_connected": bool(adj[src, dst]),
                }
            elif graph == "cora":
                N = lp_data["num_nodes"]
                if not (0 <= src < N and 0 <= dst < N) or src == dst:
                    return JSONResponse(status_code=400, content={"error": "invalid pair"})
                with torch.no_grad():
                    s = float((lp_emb[src] * lp_emb[dst]).sum())
                facts = {
                    "src_title": cora_names[src], "dst_title": cora_names[dst],
                    "cosine_score": round(s, 4),
                    "already_connected": bool(known_lp[src, dst]),
                }
            else:
                return JSONResponse(status_code=400,
                                    content={"error": "graph must be 'b2b' or 'cora'"})
        except Exception:
            return JSONResponse(status_code=400, content={"error": "bad pair"})

        prompt = (
            "You are analyzing a link-prediction model. A model predicts that "
            f"node {src} and node {dst} in the '{graph}' graph are likely related "
            f"but not explicitly connected. Evidence from the model: {json.dumps(facts)}. "
            "Explain in 3-5 plain sentences why the model finds this pair promising, "
            "which evidence matters most, and one limitation of this prediction. "
            "Be concrete; do not invent facts beyond the evidence."
        )
        body = json.dumps({
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 400, "temperature": 0.3,
        }).encode()
        try:
            req = urllib.request.Request(
                LLM_URL, data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.loads(r.read())
            text = data["choices"][0]["message"]["content"].strip()
            dt = time.perf_counter() - t0
            record("/explain", dt)
            return {"graph": graph, "src": src, "dst": dst, "facts": facts,
                    "explanation": text, "source": "local-llm",
                    "llm_model": LLM_MODEL, "latency_ms": round(dt * 1000, 2)}
        except Exception as e:
            dt = time.perf_counter() - t0
            record("/explain", dt)
            return JSONResponse(status_code=503, content={
                "graph": graph, "src": src, "dst": dst, "facts": facts,
                "explanation": None,
                "source": f"local-llm-unavailable ({type(e).__name__})",
                "latency_ms": round(dt * 1000, 2)})

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

<h2>B2B synthetic graph — top unconnected company pairs (ground-truth check)</h2>
<div class="grid" id="b2b-headline"></div>
<table id="b2b-top"><tr><th>score</th><th>company A</th><th>company B</th><th>common partners</th><th>true latent edge?</th></tr></table>

<h2>Explain a prediction (local LLM)</h2>
<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
<select id="xgraph" style="background:#0d1117;border:1px solid #30363d;color:#e6edf3;border-radius:6px;padding:7px 10px">
<option value="b2b">b2b</option><option value="cora">cora</option></select>
<input id="xsrc" type="number" value="0" style="width:90px">
<span class="dim">and</span>
<input id="xdst" type="number" value="1" style="width:90px">
<button onclick="doExplain()">Explain</button>
<span class="dim" id="x-lat"></span>
</div>
<div style="height:14px"></div>
<div id="x-out" class="card" style="white-space:pre-wrap;display:none"></div>

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
  const R=L.negative_sampling?Object.values(L.negative_sampling).join(' · '):'';
  document.getElementById('lp-metrics').innerHTML=
    card('AUC (uniform negs)',L.auc_uniform??'—')+card('MRR (uniform negs)',L.mrr_uniform??'—')+
    card('AUC (hard same-topic)',L.auc_hard??'—')+card('MRR (hard same-topic)',L.mrr_hard??'—')+
    card('Recall@10 (full ranking)',L['recall@10']??'—')+card('Recall@20 (full ranking)',L['recall@20']??'—')+
    `<div class="card"><div class="v" style="font-size:16px;padding-top:10px">${L.heldout_edges??'—'} held-out edges</div><div class="l">${L.dataset} · R@1=${L['recall@1']??'—'} R@5=${L['recall@5']??'—'} · dup pairs excluded: ${L.duplicate_feature_pairs_excluded??'—'}</div></div>`;
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
  try{
    const b=await j('/b2b');
    let bh='';
    for(const [mm,p10] of Object.entries(b.model_headline||{})){
      bh+=card(`P@10 ${mm}`,p10??'—');
    }
    document.getElementById('b2b-headline').innerHTML=bh||'<div class="card"><div class="l">b2b not trained yet</div></div>';
    let br='';
    for(const p of b.top_pairs||[]){
      br+=`<tr><td class="mono">${p.score}</td><td>${p.src} (${p.src_sector}, ${p.src_region})</td><td>${p.dst} (${p.dst_sector}, ${p.dst_region})</td><td class="mono">${p.common_neighbors}</td><td>${p.is_latent_ground_truth?'<b style="color:#3fb950">yes</b>':'no'}</td></tr>`;
    }
    document.getElementById('b2b-top').innerHTML='<tr><th>score</th><th>company A</th><th>company B</th><th>common partners</th><th>true latent edge?</th></tr>'+br;
  }catch(e){console.warn('b2b',e)}
}
async function doExplain(){
  const g=document.getElementById('xgraph').value, s=document.getElementById('xsrc').value, d=document.getElementById('xdst').value;
  const r=await fetch(`/explain?graph=${g}&src=${s}&dst=${d}`);
  const x=await r.json();
  document.getElementById('x-lat').textContent=`${r.status} · local ${x.latency_ms} ms · ${x.source}`;
  const el=document.getElementById('x-out');
  el.style.display='block';
  el.textContent = x.explanation
    ? `facts: ${JSON.stringify(x.facts)}\n\n${x.explanation}`
    : `(no LLM explanation: ${x.source})\nfacts: ${JSON.stringify(x.facts)}`;
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
