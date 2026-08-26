#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
co-embedding: 将 enhancer_seq + N分隔 + promoter_seq 拼接成一条序列, 用 Genos embed 一次。
逐对 embed (每对是独立拼接), 1024 维。带断点续跑 + 并发 + 重试。
输出: EP_ATLAS/embeddings/co_{cis,trans}_ids.npy / co_embeddings.npy
"""
import os, re, json, time
import numpy as np, pandas as pd
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT=os.path.join(os.path.dirname(__file__),"..")
API=os.path.join(ROOT,"genos","api.txt")
OUT=os.path.join(ROOT,"EP_ATLAS","embeddings")
SEP=100   # N 分隔长度
WORKERS=4; RETRY=4

def load_api():
    txt=open(API,encoding="utf-8").read()
    url=re.search(r'"(https://[^"]*dna_embedding[^"]*)"',txt).group(1)
    key=re.search(r"Bearer (sk-\S+)",txt).group(1).strip('"')
    return url,key
URL,KEY=load_api()

def embed_one(pid, seq):
    last=None
    for a in range(RETRY):
        try:
            payload={"model":"Genos","sequence":seq,"model_name":"Genos-1.2B","pooling_method":"mean"}
            req=urllib.request.Request(URL,data=json.dumps(payload).encode(),
                headers={"Content-Type":"application/json","Authorization":f"Bearer {KEY}"})
            with urllib.request.urlopen(req,timeout=600) as r: out=json.loads(r.read().decode())
            if out.get("status")!=200 or "embedding" not in out.get("result",{}): raise RuntimeError(out)
            v=out["result"]["embedding"][0]
            if len(v)!=1024: raise RuntimeError(f"dim {len(v)}")
            return pid, np.array(v,dtype=np.float32)
        except Exception as e:
            last=e; time.sleep(1.5*(a+1))
    raise RuntimeError(f"{pid}: {last}")

def run(group):
    df=pd.read_csv(os.path.join(ROOT,"EP_ATLAS",group,f"{group}_pairs_sequences.tsv"),sep='\t')
    print(f"[{group}] {len(df)} 对, 待embed",flush=True)
    # 断点
    done={}
    ip,ep=os.path.join(OUT,f"co_{group}_ids.npy"),os.path.join(OUT,f"co_{group}_embeddings.npy")
    if os.path.exists(ip) and os.path.exists(ep):
        try:
            ids=np.load(ip,allow_pickle=True).tolist(); mat=np.load(ep,allow_pickle=True)
            done={i:v for i,v in zip(ids,mat)}; print(f"  恢复{len(done)}",flush=True)
        except Exception: done={}
    # 构造 co 序列
    tasks=[]
    for i,row in df.iterrows():
        key=f"{row['enhancer_id']}__{row['promoter_id']}"
        if key in done: continue
        co=str(row['enhancer_sequence'])+"N"*SEP+str(row['promoter_sequence'])
        tasks.append((key,co))
    print(f"  待embed {len(tasks)}",flush=True)
    results=dict(done); n=0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs={ex.submit(embed_one,k,s):k for k,s in tasks}
        for fut in as_completed(futs):
            k,v=fut.result(); results[k]=v; n+=1
            if n%20==0:
                _save(group,results); print(f"  完成 {n}/{len(tasks)}",flush=True)
    _save(group,results); print(f"[{group}] 全部完成 {len(results)}",flush=True)

def _save(group,results):
    os.makedirs(OUT,exist_ok=True)
    order=sorted(results.keys())
    np.save(os.path.join(OUT,f"co_{group}_ids.npy"),np.array(order,dtype=object))
    np.save(os.path.join(OUT,f"co_{group}_embeddings.npy"),np.stack([results[i] for i in order]))

if __name__=="__main__":
    import sys
    run(sys.argv[1] if len(sys.argv)>1 else "cis")
