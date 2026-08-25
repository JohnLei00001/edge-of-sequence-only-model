#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 EP_ATLAS K562 CSV 提取增强子/启动子序列（参考基因组 hg19）。
- 按 enhancer_id / promoter_id 去重（同一ID坐标应一致），提取唯一序列
- 统一大写
- 输出：enhancers.fa, promoters.fa, 及完整互作关联表
"""
import os, csv, sys
from collections import OrderedDict

ROOT = os.path.join(os.path.dirname(__file__), "..")
IN   = os.path.join(ROOT, "EP_ATLAS", "K562_Genos_P001_read2_all_methods.csv")
HG19 = os.path.join(ROOT, "references", "hg19", "hg19.fa")
OUT  = os.path.join(ROOT, "EP_ATLAS", "sequences")

def main():
    from pyfaidx import Fasta
    fa = Fasta(HG19)
    rows = list(csv.DictReader(open(IN, encoding="utf-8")))

    enhancers = OrderedDict()   # id -> dict(chrom,start,end,coords_tuple)
    promoters = OrderedDict()
    for r in rows:
        e = r["enhancer_id"]
        enhancers.setdefault(e, (r["enhancer_chr"], int(r["enhancer_chr_start"]),
                                 int(r["enhancer_chr_end"])))
        p = r["promoter_id"]
        promoters.setdefault(p, (r["promoter_chr"], int(r["promoter_chr_start"]),
                                 int(r["promoter_chr_end"])))

    def extract(coords):
        chrom, s, e = coords
        if chrom not in fa or not (1 <= s <= e <= len(fa[chrom])):
            return None, f"out_of_range:{chrom}:{s}-{e}"
        return str(fa.get_seq(chrom, s, e)).upper(), "OK"

    os.makedirs(OUT, exist_ok=True)
    enh_fa, prom_fa, enh_status, prom_status = [], [], {}, {}
    enh_bad = prom_bad = 0
    for eid, coords in enhancers.items():
        seq, st = extract(coords)
        enh_status[eid] = st
        if seq is not None:
            enh_fa.append((f"{eid}|{coords[0]}:{coords[1]}-{coords[2]}", seq))
        else:
            enh_bad += 1
    for pid, coords in promoters.items():
        seq, st = extract(coords)
        prom_status[pid] = st
        if seq is not None:
            prom_fa.append((f"{pid}|{coords[0]}:{coords[1]}-{coords[2]}", seq))
        else:
            prom_bad += 1

    def wfa(path, recs):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            for sid, seq in recs:
                f.write(f">{sid}\n")
                for i in range(0, len(seq), 80):
                    f.write(seq[i:i+80] + "\n")
    wfa(os.path.join(OUT, "enhancers.fa"), enh_fa)
    wfa(os.path.join(OUT, "promoters.fa"), prom_fa)

    # 互作关联表
    with open(os.path.join(OUT, "enhancer_promoter_linkage.tsv"),
              "w", encoding="utf-8", newline="\n") as f:
        f.write("enhancer_id\tenh_chrom\tenh_start\tenh_end\tenh_seq_id\t"
                "promoter_id\tprom_chrom\tprom_start\tprom_end\tprom_seq_id\t"
                "ENSG\tgene_id\tgene_type\treads_count\tP-value\tmethods\t"
                "enh_status\tprom_status\n")
        for r in rows:
            e, p = r["enhancer_id"], r["promoter_id"]
            ec, es, ee = enhancers[e]
            pc, ps, pe = promoters[p]
            line = "\t".join(map(str, (
                e, ec, es, ee, f"{e}|{ec}:{es}-{ee}",
                p, pc, ps, pe, f"{p}|{pc}:{ps}-{pe}",
                r["ENSG"], r["gene_id"], r["gene_type"],
                r["reads_count"], r["P-value"], r["methods"],
                enh_status[e], prom_status[p])))
            f.write(line + "\n")

    print(f"互作记录: {len(rows)}  唯一增强子: {len(enhancers)}  唯一启动子: {len(promoters)}")
    print(f"增强子序列提取: {len(enh_fa)}  失败: {enh_bad}")
    print(f"启动子序列提取: {len(prom_fa)}  失败: {prom_bad}")
    if enh_bad:
        print("  增强子失败示例:", [x for x, v in enh_status.items() if v != "OK"][:5])
    if prom_bad:
        print("  启动子失败示例:", [x for x, v in prom_status.items() if v != "OK"][:5])

if __name__ == "__main__":
    main()
