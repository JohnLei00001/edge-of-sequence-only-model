#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 ENdb 增强子坐标提取两类序列：
  1. 增强子序列   —— 按 [Start,End] 从对应参考基因组直接切片
  2. 靶基因启动子 —— 按基因规范TSS，取 TSS-2000 ~ TSS+500（按链方向），
                     序列按基因编码链方向（'-'链反向互补）

坐标系约定：ENdb 与 TSS 索引均为 1-based 全闭区间；pyfaidx fetch 用 1-based 全闭。
参考基因组来源：UCSC 各 build 的 .fa.gz（与 ENdb 染色体命名 chr* 一致）。
"""
import os, sys, re
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..")
COORDS = os.path.join(ROOT, "ENdb_data", "enhancer_coords.tsv")
TSSIDX = os.path.join(ROOT, "references", "annot", "tss_index.tsv")
REFS   = os.path.join(ROOT, "references")
OUTDIR = os.path.join(ROOT, "sequences")
PROMPT_UP   = int(os.environ.get("PROMPT_UP", "2000"))   # TSS 上游
PROMPT_DOWN = int(os.environ.get("PROMPT_DOWN", "500"))  # TSS 下游

BUILD2FA = {
    "hg19":     os.path.join(REFS, "hg19", "hg19.fa"),
    "hg38":     os.path.join(REFS, "hg38", "hg38.fa"),
    "mm10":     os.path.join(REFS, "mm10", "mm10.fa"),
    "mm39":     os.path.join(REFS, "mm39", "mm39.fa"),
    "danRer11": os.path.join(REFS, "danRer11", "danRer11.fa"),
    "dm6":      os.path.join(REFS, "dm6", "dm6.fa"),
    "galGal6":  os.path.join(REFS, "galGal6", "galGal6.fa"),
    "GRCz11":   os.path.join(REFS, "danRer11", "danRer11.fa"),  # ENdb 版本别名
}

def normalize_build(b):
    """统一基因组版本命名（ENdb 的 GRCz11 == danRer11）。"""
    return {"GRCz11": "danRer11"}.get(b, b)

# 常见别名（键=大写形式）。注意：部分为物种特异（如 EVI1->Mecom 在小鼠），
# 此处统一按大写匹配，由注释索引按 build 决定最终基因。
ALIAS = {
    "EVI1": "MECOM", "EVI-1": "MECOM",
    "RANKL": "TNFSF11", "MCK": "CKM", "CJUN": "JUN",
    "M-CREB": "CREB1", "GABP": "GABPA", "CJUN": "JUN",
}

def clean_gene(g):
    """清洗基因符号：去空白、去括号别名后缀（如 'Tnfsf11 (Rankl)' -> 'Tnfsf11'），别名归一。"""
    g = g.strip()
    m = re.match(r"^(.+?)\s*\([^)]*\)\s*$", g)
    if m:
        g = m.group(1).strip()
    return ALIAS.get(g.upper(), g)

def load_coords():
    rows = []
    with open(COORDS, encoding="utf-8") as f:
        f.readline()
        for line in f:
            c = line.rstrip("\n").split("\t")
            rows.append(dict(enh_id=c[0], build=c[1], chrom=c[2],
                             start=int(c[3]), end=int(c[4]),
                             species=c[5], genes=[g for g in c[6].split(",") if g],
                             year=c[7], pmid=c[8], etype=c[9]))
    return rows

def load_tss():
    idx = defaultdict(list)  # (build,gene_upper) -> [(gene_symbol, chrom, strand, tss), ...]
    if not os.path.exists(TSSIDX):
        return idx
    with open(TSSIDX, encoding="utf-8") as f:
        f.readline()
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 6:
                continue
            build, gene, gup = c[0], c[1], c[2]
            chrom, tss, strand = c[3], int(c[4]), c[5]
            idx[(build, gup)].append((gene, chrom, strand, tss))
    return idx

def write_fasta(path, recs):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for seqid, seq in recs:
            f.write(f">{seqid}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")

def main():
    from pyfaidx import Fasta
    coords = load_coords()
    tss = load_tss()
    os.makedirs(OUTDIR, exist_ok=True)

    fa_cache = {}
    enh_fa, prom_fa, linkage = [], [], []
    stats = defaultdict(int)

    for rec in coords:
        build = normalize_build(rec["build"])
        rec["build"] = build
        gstr = ",".join(rec["genes"])
        chrom, s, e = rec["chrom"], rec["start"], rec["end"]
        fa_path = BUILD2FA.get(build)
        if not fa_path or not os.path.exists(fa_path):
            stats["skip_no_genome"] += 1
            linkage.append([rec["enh_id"], build, chrom, s, e, gstr,
                            "", "", "", "", "", "", "", "no_genome_file", ""])
            continue
        if build not in fa_cache:
            fa_cache[build] = Fasta(fa_path)
        fa = fa_cache[build]
        chrom_len = len(fa[chrom]) if chrom in fa else None
        if chrom not in fa:
            stats["enh_chrom_not_found"] += 1
            linkage.append([rec["enh_id"], build, chrom, s, e, gstr,
                            "", "", "", "", "", "", "", "chrom_not_in_genome", ""])
            continue

        # ---- 增强子序列 ----
        eid = f"{rec['enh_id']}|{build}:{chrom}:{s}-{e}"
        enh_ok = False
        if 1 <= s <= e <= chrom_len:
            enh_seq = str(fa.get_seq(chrom, s, e)).upper()
            enh_fa.append((eid, enh_seq))
            enh_ok = True
            stats["enh_extracted"] += 1
        else:
            stats["enh_out_of_range"] += 1
            linkage.append([rec["enh_id"], build, chrom, s, e, gstr,
                            "", "", "", "", "", "", "", "enhancer_out_of_range", ""])

        # ---- 靶基因启动子 ----
        if not rec["genes"]:
            stats["promo_no_gene"] += 1
            if enh_ok:
                linkage.append([rec["enh_id"], build, chrom, s, e, "", "", "", "",
                                "", "", "", "", "OK", "no_target_gene"])
            continue
        for gene in rec["genes"]:
            cand = tss.get((build, clean_gene(gene).upper()))
            if not cand:
                stats["promo_gene_not_in_annotation"] += 1
                if enh_ok:
                    linkage.append([rec["enh_id"], build, chrom, s, e, gene,
                                    "", "", "", "", "", "", "", "OK", "gene_not_in_annotation"])
                continue
            # 一个基因可能对应多个TSS（罕见），逐条处理
            for (g_symbol, gchrom, gstrand, gtss) in cand:
                if gchrom != chrom:
                    # 基因TSS与增强子不同染色体：记录但不提取（跨物种/注释不一致）
                    stats["promo_gene_chrom_mismatch"] += 1
                    linkage.append([rec["enh_id"], build, chrom, s, e, g_symbol,
                                    gchrom, gstrand, gtss, "", "", "", "",
                                    "OK" if enh_ok else "enh_only", "gene_chrom_mismatch"])
                    continue
                if gstrand == "+":
                    pstart, pend = gtss - PROMPT_UP, gtss + PROMPT_DOWN
                    pstrand = "+"
                else:
                    pstart, pend = gtss - PROMPT_DOWN, gtss + PROMPT_UP
                    pstrand = "-"
                if pstart < 1: pstart = 1
                if pend > chrom_len: pend = chrom_len
                if pstart > pend:
                    stats["promo_out_of_range"] += 1
                    continue
                pid = (f"{rec['enh_id']}|{g_symbol}|{build}:{chrom}:{pstart}-{pend}"
                       f"[{gstrand}]")
                pseq = str(fa.get_seq(chrom, pstart, pend, rc=(pstrand == "-"))).upper()
                prom_fa.append((pid, pseq))
                stats["promo_extracted"] += 1
                linkage.append([rec["enh_id"], build, chrom, s, e, g_symbol,
                                gchrom, gstrand, gtss, pstart, pend, pstrand, pid,
                                "OK" if enh_ok else "enh_only", "OK"])

    write_fasta(os.path.join(OUTDIR, "enhancers.fa"), enh_fa)
    write_fasta(os.path.join(OUTDIR, "promoters.fa"), prom_fa)
    with open(os.path.join(OUTDIR, "enhancer_promoter_linkage.tsv"),
              "w", encoding="utf-8", newline="\n") as f:
        f.write("enh_id\tbuild\tenh_chrom\tenh_start\tenh_end\tgene\t"
                "gene_chrom\tgene_strand\ttss\tprom_start\tprom_end\tprom_strand\t"
                "prom_seq_id\tenhancer_seq_status\tpromoter_seq_status\n")
        for r in linkage:
            f.write("\t".join(map(str, r)) + "\n")

    print("=== 统计 ===")
    for k in sorted(stats):
        print(f"  {k}: {stats[k]}")
    print(f"  增强子FASTA: {len(enh_fa)} 条")
    print(f"  启动子FASTA: {len(prom_fa)} 条")

if __name__ == "__main__":
    main()
