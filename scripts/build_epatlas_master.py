#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
归总 EP_ATLAS 数据库：把互作关联表与序列合并为自包含主表。
主表每条互作一行，内联增强子/启动子序列 + 全部元数据，供 sequence-only 模型直接使用。
"""
import os, csv

ROOT = os.path.join(os.path.dirname(__file__), "..")
SEQ  = os.path.join(ROOT, "EP_ATLAS", "sequences")
DB   = os.path.join(ROOT, "EP_ATLAS", "database")
LINK = os.path.join(SEQ, "enhancer_promoter_linkage.tsv")

def read_fa(path):
    d = {}
    n = None
    for l in open(path, encoding="utf-8"):
        l = l.rstrip("\n")
        if l.startswith(">"):
            n = l[1:]; d[n] = []
        elif n is not None:
            d[n].append(l)
    return {k: "".join(v) for k, v in d.items()}

def main():
    enh_fa = read_fa(os.path.join(SEQ, "enhancers.fa"))
    prom_fa = read_fa(os.path.join(SEQ, "promoters.fa"))
    os.makedirs(DB, exist_ok=True)

    out = os.path.join(DB, "EP_ATLAS_master.tsv")
    with open(LINK, encoding="utf-8") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        with open(out, "w", encoding="utf-8", newline="\n") as fo:
            fo.write("enhancer_id\tenh_chrom\tenh_start\tenh_end\tenh_length\t"
                     "enhancer_sequence\t"
                     "promoter_id\tprom_chrom\tprom_start\tprom_end\tprom_length\t"
                     "promoter_sequence\t"
                     "ENSG\tgene_id\tgene_type\treads_count\tP-value\tcell_line\tmethods\n")
            n = 0
            for r in rdr:
                enh = enh_fa.get(r["enh_seq_id"], "")
                prom = prom_fa.get(r["prom_seq_id"], "")
                fo.write("\t".join([
                    r["enhancer_id"], r["enh_chrom"], r["enh_start"], r["enh_end"],
                    str(len(enh)), enh,
                    r["promoter_id"], r["prom_chrom"], r["prom_start"], r["prom_end"],
                    str(len(prom)), prom,
                    r["ENSG"], r["gene_id"], r["gene_type"],
                    r["reads_count"], r["P-value"], "K562", r["methods"],
                ]) + "\n")
                n += 1
    print(f"主表已生成: {out}, 互作记录 {n} 条")
    print(f"增强子序列 {len(enh_fa)}  启动子序列 {len(prom_fa)}")

if __name__ == "__main__":
    main()
