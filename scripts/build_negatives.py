#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建平衡 cis/trans 数据集的负样本（打乱配对 + 全集过滤 + 分层抽样）。

逻辑：
  cis   负样本 = cis 池(E) × cis 池(P) 中"同染色体"组合，
                剔除 K562_download.txt 全集里有证据的对，
                按正样本 cis 距离分布【分箱分层抽样】1263 条。
  trans 负样本 = trans 池(E) × trans 池(P) 中"跨染色体"组合，
                剔除全集中有证据的对，随机抽样 1263 条。

输出(合并正负, label=1/0, 并做按增强子无泄漏切分 70/15/15):
  EP_ATLAS/cis/cis_pairs_labeled.tsv
  EP_ATLAS/trans/trans_pairs_labeled.tsv
  EP_ATLAS/negative_sampling_report.md
"""
import os, csv, math, random
from collections import Counter, defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..")
BAL  = os.path.join(ROOT, "EP_ATLAS", "K562_Genos_balanced_cis_trans_2526_pairs.csv")
FULL = os.path.join(ROOT, "EP_ATLAS", "K562_download.txt")
CIS  = os.path.join(ROOT, "EP_ATLAS", "cis")
TRANS= os.path.join(ROOT, "EP_ATLAS", "trans")
REPORT= os.path.join(ROOT, "EP_ATLAS", "negative_sampling_report.md")
SEED = 42
NEG_PER_GROUP = 1263   # 与正样本等量

def decade_bin(d):
    if d <= 0:
        return "0"
    k = int(math.floor(math.log10(d)))
    return f"1e{k}-1e{k+1}"

def midpoint(s, e):
    return (int(s) + int(e)) / 2.0

def main():
    random.seed(SEED)
    pos = list(csv.DictReader(open(BAL, encoding="utf-8")))
    full = list(csv.DictReader(open(FULL, encoding="utf-8"), delimiter="\t"))
    full_pairs = set((r["enhancer_id"], r["promoter_id"]) for r in full)
    print(f"正样本: {len(pos)}   全集互作对: {len(full_pairs)}")

    # ---- 构建 cis/trans 池(E/P + 坐标 + 所属基因) ----
    pools = {g: {"E": {}, "P": {}} for g in ("cis", "trans")}
    for r in pos:
        g = "cis" if r["interaction_type"] == "cis_same_chromosome" else "trans"
        pools[g]["E"].setdefault(r["enhancer_id"],
            (r["enhancer_chr"], int(r["enhancer_chr_start"]), int(r["enhancer_chr_end"])))
        pools[g]["P"].setdefault(r["promoter_id"],
            (r["promoter_chr"], int(r["promoter_chr_start"]), int(r["promoter_chr_end"]),
             r["gene_id"], r["gene_type"]))

    # ---- 生成候选负样本 ----
    # cis: 同染色体; trans: 跨染色体; 均剔除全集中存在的对
    candidates = {"cis": [], "trans": []}   # (E_id, P_id, dist) dist None for trans
    pos_cis_dists = []
    for r in pos:
        if r["interaction_type"] == "cis_same_chromosome":
            d = abs(midpoint(r["promoter_chr_start"], r["promoter_chr_end"])
                    - midpoint(r["enhancer_chr_start"], r["enhancer_chr_end"]))
            pos_cis_dists.append(d)
    for g in ("cis", "trans"):
        E = pools[g]["E"]; P = pools[g]["P"]
        same = (g == "cis")
        for eid, (ec, es, ee) in E.items():
            for pid, (pc, ps, pe, gid, gtype) in P.items():
                is_same = (ec == pc)
                if is_same != same:
                    continue  # 染色体约束
                if (eid, pid) in full_pairs:
                    continue  # 全集有证据, 剔除
                if same:
                    d = abs(midpoint(ps, pe) - midpoint(es, ee))
                    candidates[g].append((eid, pid, d, gid, gtype))
                else:
                    candidates[g].append((eid, pid, None, gid, gtype))
        print(f"{g} 候选负样本(剔除全集后): {len(candidates[g]):,}")

    # ---- 抽样 ----
    # cis: 按正样本距离分布分箱分层抽样
    pos_bins = Counter(decade_bin(d) for d in pos_cis_dists)
    total_pos = sum(pos_bins.values())
    by_bin = defaultdict(list)
    for cand in candidates["cis"]:
        by_bin[decade_bin(cand[2])].append(cand)
    # 分配各箱目标
    alloc = {b: round(NEG_PER_GROUP * n / total_pos) for b, n in pos_bins.items()}
    selected = []
    for b, target in alloc.items():
        cands = by_bin.get(b, [])
        random.shuffle(cands)
        selected.extend(cands[:min(target, len(cands))])
    # 若不足, 从其余候选补足(优先补最接近目标比例的箱)
    if len(selected) < NEG_PER_GROUP:
        picked = {c[:2] for c in selected}
        remain = [c for b, cs in by_bin.items() for c in cs if c[:2] not in picked]
        random.shuffle(remain)
        selected.extend(remain[:NEG_PER_GROUP - len(selected)])
    cis_neg = selected[:NEG_PER_GROUP]

    # trans: 随机抽样
    random.shuffle(candidates["trans"])
    trans_neg = candidates["trans"][:NEG_PER_GROUP]
    print(f"抽样: cis负 {len(cis_neg)}  trans负 {len(trans_neg)}")

    # ---- 合并正负, 打标签, 按增强子无泄漏切分 ----
    def build_labeled(g, neg):
        out = []
        # 正样本
        for r in pos:
            if (r["interaction_type"] == "cis_same_chromosome") != (g == "cis"):
                continue
            d = abs(midpoint(r["promoter_chr_start"], r["promoter_chr_end"])
                    - midpoint(r["enhancer_chr_start"], r["enhancer_chr_end"]))
            out.append([r["enhancer_id"], r["enhancer_chr"], r["enhancer_chr_start"],
                        r["enhancer_chr_end"], r["promoter_id"], r["promoter_chr"],
                        r["promoter_chr_start"], r["promoter_chr_end"], r["gene_id"],
                        r["gene_type"], r["reads_count"], r["P-value"], r["methods"],
                        r["interaction_type"], decade_bin(d) if g == "cis" else "trans",
                        1])
        # 负样本
        for (eid, pid, dist, gid, gtype) in neg:
            ec, es, ee = pools[g]["E"][eid]
            pc, ps, pe, _, _ = pools[g]["P"][pid]
            out.append([eid, ec, es, ee, pid, pc, ps, pe, gid, gtype,
                        "", "", "", "cis_same_chromosome" if g == "cis" else "trans_cross_chromosome",
                        decade_bin(dist) if g == "cis" else "trans", 0])
        # 按增强子无泄漏切分 70/15/15
        enhancers = list({r[0] for r in out})
        random.seed(SEED)
        random.shuffle(enhancers)
        n = len(enhancers)
        ntr, nva = int(n*0.7), int(n*0.15)
        fold_of = {}
        for i, e in enumerate(enhancers):
            fold_of[e] = "train" if i < ntr else ("val" if i < ntr+nva else "test")
        for r in out:
            r.append(fold_of[r[0]])
        return out

    cols = ["enhancer_id", "enhancer_chr", "enhancer_start", "enhancer_end",
            "promoter_id", "promoter_chr", "promoter_start", "promoter_end",
            "gene_id", "gene_type", "reads_count", "P-value", "methods",
            "interaction_type", "distance_bin", "label", "fold"]

    cis_out = build_labeled("cis", cis_neg)
    trans_out = build_labeled("trans", trans_neg)
    for g, out, path in [("cis", cis_out, os.path.join(CIS, "cis_pairs_labeled.tsv")),
                         ("trans", trans_out, os.path.join(TRANS, "trans_pairs_labeled.tsv"))]:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\t".join(cols) + "\n")
            for r in out:
                f.write("\t".join(map(str, r)) + "\n")
        print(f"写出 {path}: {len(out)} 行")
    print(f"负样本数: cis {len(cis_neg)} + trans {len(trans_neg)} = {len(cis_neg)+len(trans_neg)}")

if __name__ == "__main__":
    main()
