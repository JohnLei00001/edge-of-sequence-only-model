#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全量调用 Genos dna_embedding 为唯一增强子/启动子生成 1024 维 embedding。
- 从 genos/api.txt 读取端点与密钥(本地, 不入库)
- 并发线程 + 重试 + 断点续跑
- 输出: EP_ATLAS/embeddings/{enhancers,promoters}_ids.npy 与 _embeddings.npy
"""
import os, re, json, time, math, threading
import urllib.request, urllib.error
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.join(os.path.dirname(__file__), "..")
API  = os.path.join(ROOT, "genos", "api.txt")
FA   = os.path.join(ROOT, "EP_ATLAS", "sequences")
OUT  = os.path.join(ROOT, "EP_ATLAS", "embeddings")
WORKERS = int(os.environ.get("EMBED_WORKERS", "5"))
RETRY   = int(os.environ.get("EMBED_RETRY", "4"))

def load_api():
    txt = open(API, encoding="utf-8").read()
    url = re.search(r'"(https://[^"]*dna_embedding[^"]*)"', txt).group(1)
    key = re.search(r"Bearer (sk-\S+)", txt).group(1).strip('"')
    return url, key

def read_fa(path):
    recs, name = {}, None
    for l in open(path, encoding="utf-8"):
        l = l.rstrip("\n")
        if l.startswith(">"):
            name = l[1:].split("|")[0]
            recs[name] = []
        elif name is not None:
            recs[name].append(l)
    return {k: "".join(v) for k, v in recs.items()}

URL, KEY = load_api()

def embed_one(seq_id, seq, url=URL, key=KEY):
    last = None
    for attempt in range(RETRY):
        try:
            payload = {"model": "Genos", "sequence": seq,
                       "model_name": "Genos-1.2B", "pooling_method": "mean"}
            req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=300) as r:
                out = json.loads(r.read().decode())
            if out.get("status") != 200 or "embedding" not in out.get("result", {}):
                raise RuntimeError(f"bad resp: {json.dumps(out)[:120]}")
            vec = out["result"]["embedding"][0]
            if len(vec) != 1024:
                raise RuntimeError(f"dim {len(vec)}")
            return seq_id, np.array(vec, dtype=np.float32)
        except Exception as e:
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{seq_id}: {last}")

def run_group(name, fasta, ids_path, emb_path):
    seqs = read_fa(fasta)
    print(f"[{name}] 待embed: {len(seqs)}", flush=True)
    # 断点: 已完成的id
    done = {}
    if os.path.exists(emb_path) and os.path.exists(ids_path):
        try:
            old_ids = np.load(ids_path, allow_pickle=True).tolist()
            old_emb = np.load(emb_path, allow_pickle=True)
            done = {i: v for i, v in zip(old_ids, old_emb)}
            print(f"[{name}] 恢复 {len(done)} 条", flush=True)
        except Exception:
            done = {}
    todo = [i for i in seqs if i not in done]
    results = dict(done)
    print(f"[{name}] 待完成 {len(todo)}", flush=True)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(embed_one, i, seqs[i]): i for i in todo}
        n = 0
        for fut in as_completed(futs):
            i, v = fut.result()
            results[i] = v
            n += 1
            if n % 50 == 0:
                _save(name, results)
                print(f"[{name}] 完成 {n}/{len(todo)}", flush=True)
    _save(name, results)
    print(f"[{name}] 全部完成 {len(results)}", flush=True)

def _save(name, results):
    os.makedirs(OUT, exist_ok=True)
    order = sorted(results.keys())
    ids = np.array(order, dtype=object)
    mat = np.stack([results[i] for i in order])
    np.save(os.path.join(OUT, f"{name}_ids.npy"), ids)
    np.save(os.path.join(OUT, f"{name}_embeddings.npy"), mat)

if __name__ == "__main__":
    run_group("enhancers", os.path.join(FA, "enhancers.fa"),
              os.path.join(OUT, "enhancers_ids.npy"),
              os.path.join(OUT, "enhancers_embeddings.npy"))
    run_group("promoters", os.path.join(FA, "promoters.fa"),
              os.path.join(OUT, "promoters_ids.npy"),
              os.path.join(OUT, "promoters_embeddings.npy"))
    print("DONE")
