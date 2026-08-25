#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从平衡的 cis/trans 数据集(K562_Genos_balanced_cis_trans_2526_pairs.csv)，
按 interaction_type 拆分为 cis / trans 两组，分别用 hg19 提取增强子/启动子序列，分目录存放。

输出：
  EP_ATLAS/cis/
    pairs.tsv       cis 子集配对表
    enhancers.fa    唯一增强子序列
    promoters.fa    唯一启动子序列
  EP_ATLAS/trans/   (同上)
"""
import os, csv
from collections import OrderedDict

ROOT = os.path.join(os.path.dirname(__file__), "..")
IN   = os.path.join(ROOT, "EP_ATLAS", "K562_Genos_balanced_cis_trans_2526_pairs.csv")
HG19 = os.path.join(ROOT, "references", "hg19", "hg19.fa")
GROUPS = {
    "cis_same_chromosome":   os.path.join(ROOT, "EP_ATLAS", "cis"),
    "trans_cross_chromosome": os.path.join(ROOT, "EP_ATLAS", "trans"),
}

def read_fa_dict(p):
    return None

def main():
    from pyfaidx import Fasta
    fa = Fasta(HG19)
    rows = list(csv.DictReader(open(IN, encoding="utf-8")))

    bygroup = {g: [] for g in GROUPS}
    for r in rows:
        t = r["interaction_type"]
        if t in bygroup:
            bygroup[t].append(r)

    def extract(coords):
        chrom, s, e = coords
        if chrom not in fa or not (1 <= s <= e <= len(fa[chrom])):
            return None
        return str(fa.get_seq(chrom, s, e)).upper()

    for g, outdir in GROUPS.items():
        grp = bygroup[g]
        os.makedirs(outdir, exist_ok=True)
        enh = OrderedDict()
        prom = OrderedDict()
        bad_e = bad_p = 0
        for r in grp:
            eid = r["enhancer_id"]
            enh.setdefault(eid, (r["enhancer_chr"], int(r["enhancer_chr_start"]),
                                 int(r["enhancer_chr_end"])))
            pid = r["promoter_id"]
            prom.setdefault(pid, (r["promoter_chr"], int(r["promoter_chr_start"]),
                                  int(r["promoter_chr_end"])))
        enh_fa, prom_fa = [], []
        for eid, coords in enh.items():
            s = extract(coords)
            if s is None:
                bad_e += 1
            else:
                enh_fa.append((f"{eid}|{coords[0]}:{coords[1]}-{coords[2]}", s))
        for pid, coords in prom.items():
            s = extract(coords)
            if s is None:
                bad_p += 1
            else:
                prom_fa.append((f"{pid}|{coords[0]}:{coords[1]}-{coords[2]}", s))

        def wfa(path, recs):
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                for sid, seq in recs:
                    f.write(f">{sid}\n")
                    for i in range(0, len(seq), 80):
                        f.write(seq[i:i+80] + "\n")
        wfa(os.path.join(outdir, "enhancers.fa"), enh_fa)
        wfa(os.path.join(outdir, "promoters.fa"), prom_fa)
        with open(os.path.join(outdir, "pairs.tsv"), "w", encoding="utf-8", newline="\n") as f:
            f.write("\t".join(rows[0].keys()) + "\n")
            for r in grp:
                f.write("\t".join(r.values()) + "\n")
        print(f"[{g}] 配对 {len(grp)} | 唯一增强子 {len(enh)} 提取 {len(enh_fa)} 失败{bad_e}"
              f" | 唯一启动子 {len(prom)} 提取 {len(prom_fa)} 失败{bad_p}")

if __name__ == "__main__":
    main()
