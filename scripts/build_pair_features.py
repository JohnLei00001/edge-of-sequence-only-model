#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
组装配对特征向量: 每条配对 = concat(enh_emb, prom_emb, 余弦相似度) → (2049,)
对 cis/trans 的正负样本, 连同 label / fold / 元数据保存。
"""
import os, csv, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
EMB  = os.path.join(ROOT, "EP_ATLAS", "embeddings")
OUT  = os.path.join(ROOT, "EP_ATLAS", "features")
GROUPS = {"cis", "trans"}

def load_emb(name):
    ids = np.load(os.path.join(EMB, f"{name}_ids.npy"), allow_pickle=True).tolist()
    mat = np.load(os.path.join(EMB, f"{name}_embeddings.npy"), allow_pickle=True)
    return dict(zip(ids, mat))

def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb + 1e-12))

def main():
    enh = load_emb("enhancers")
    prom = load_emb("promoters")
    os.makedirs(OUT, exist_ok=True)
    for g in GROUPS:
        rows = list(csv.DictReader(open(os.path.join(ROOT, "EP_ATLAS", g, f"{g}_pairs_labeled.tsv")),
                                   delimiter="\t"))
        X, y, metas = [], [], []
        miss = 0
        for r in rows:
            e = enh.get(r["enhancer_id"]); p = prom.get(r["promoter_id"])
            if e is None or p is None:
                miss += 1
                continue
            cos = cosine(e, p)
            vec = np.concatenate([e, p, [cos]]).astype(np.float32)
            X.append(vec); y.append(int(r["label"]))
            metas.append((r["enhancer_id"], r["promoter_id"], r["fold"],
                          r["interaction_type"], r.get("distance_bin", "")))
        X = np.stack(X); y = np.array(y)
        np.save(os.path.join(OUT, f"{g}_X.npy"), X)
        np.save(os.path.join(OUT, f"{g}_y.npy"), y)
        with open(os.path.join(OUT, f"{g}_meta.tsv"), "w", encoding="utf-8", newline="\n") as f:
            f.write("enhancer_id\tpromoter_id\tfold\tinteraction_type\tdistance_bin\n")
            for m in metas:
                f.write("\t".join(map(str, m)) + "\n")
        print(f"{g}: 配对{len(rows)} 组装{len(X)} 缺失{miss} 特征shape={X.shape} 标签正/负={sum(y==1)}/{sum(y==0)}")
    print("DONE")

if __name__ == "__main__":
    main()
