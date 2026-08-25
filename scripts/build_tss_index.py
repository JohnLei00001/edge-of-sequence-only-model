#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建 基因symbol -> 规范TSS 索引。

来源：
- 人类/小鼠：GENCODE GTF（hg19 v19, hg38 v44, mm10 M25, mm39 M35）
  - 每个 gene 取 "basic"/"level_1" 或最长转录本作为规范转录本，TSS 按链方向。
- 次要物种(dm6/danRer11/galGal6)：UCSC refGene（txStart/txEnd + strand，最长转录本）

输出：TSV 每行  build  gene_symbol  chrom  tss  strand
TSS 定义：转录本起始点（+链=txStart，-链=txEnd），坐标1-based。
"""
import os, gzip, sys
from collections import defaultdict

BASE = os.path.join(os.path.dirname(__file__), "..", "references", "annot")

def parse_gencode(path):
    """yield (gene_symbol, chrom, strand, txStart1based, txEnd1based, txLen, level, tags)"""
    recs = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) < 9 or c[2] != "transcript":
                continue
            chrom, strand = c[0], c[6]
            try:
                start, end = int(c[3]), int(c[4])
            except ValueError:
                continue
            attrs = dict(kv.strip().split(" ", 1) for kv in
                         [x for x in c[8].split(";") if x.strip()])
            def attr(k, default=""):
                v = attrs.get(k, default)
                return v.strip('"')
            gene = attr("gene_name")
            tx_id = attr("transcript_id")
            level = attr("level", "")
            tags = attr("tag", "")
            if not gene:
                continue
            recs.append(dict(gene=gene, chrom=chrom, strand=strand,
                             txStart=start, txEnd=end,
                             txLen=end - start + 1, level=level, tags=tags))
    # 每个基因选规范转录本：优先 level_1/basic，否则最长
    bygene = defaultdict(list)
    for r in recs:
        bygene[r["gene"]].append(r)
    out = []
    for gene, lst in bygene.items():
        # 排序：level_1(数字越小越规范) 优先；其次最长转录本
        def level_key(r):
            try:
                return int(r["level"])
            except (ValueError, TypeError):
                return 99
        lst_sorted = sorted(lst, key=lambda r: (level_key(r), -r["txLen"]))
        best = lst_sorted[0]
        tss = best["txStart"] if best["strand"] == "+" else best["txEnd"]
        out.append((gene, best["chrom"], best["strand"], tss))
    return out

def parse_refgene(path):
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) < 13:
                continue
            chrom, strand = c[2], c[3]
            try:
                txStart, txEnd = int(c[4]), int(c[5])
            except ValueError:
                continue
            gene = c[12]
            if not gene or gene == ".":
                continue
            out.append((gene, chrom, strand, txStart, txEnd))
    # 每个基因最长转录本
    bygene = defaultdict(list)
    for g, chrom, strand, s, e in out:
        bygene[g].append((chrom, strand, s, e))
    res = []
    for g, lst in bygene.items():
        lst_sorted = sorted(lst, key=lambda x: -(x[3] - x[2] + 1))
        chrom, strand, s, e = lst_sorted[0]
        tss = s if strand == "+" else e
        res.append((g, chrom, strand, tss))
    return res

def main():
    jobs = {
        # build -> (type, filename)
        "hg19":    ("gencode", "gencode.v19.hg19.gtf.gz"),
        "hg38":    ("gencode", "gencode.v44.hg38.gtf.gz"),
        "mm10":    ("gencode", "gencode.M25.mm10.gtf.gz"),
        "mm39":    ("gencode", "gencode.M35.mm39.gtf.gz"),
        "dm6":     ("refgene", "refGene.dm6.txt.gz"),
        "danRer11":("refgene", "refGene.danRer11.txt.gz"),
        "galGal6": ("refgene", "refGene.galGal6.txt.gz"),
    }
    out_path = os.path.join(BASE, "tss_index.tsv")
    all_recs = []
    for build, (kind, fn) in jobs.items():
        p = os.path.join(BASE, fn)
        if not os.path.exists(p):
            print(f"[跳过] {build}: 缺少 {fn}", file=sys.stderr)
            continue
        recs = parse_gencode(p) if kind == "gencode" else parse_refgene(p)
        print(f"[OK] {build}: {len(recs)} 基因")
        for g, chrom, strand, tss in recs:
            all_recs.append((build, g, g.upper(), chrom, tss, strand))
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("build\tgene_symbol\tgene_upper\tchrom\ttss\tstrand\n")
        for build, g, gup, chrom, tss, strand in all_recs:
            f.write("\t".join(map(str, (build, g, gup, chrom, tss, strand))) + "\n")
    print(f"\nTSS 索引已写入 {out_path}，共 {len(all_recs)} 条")

if __name__ == "__main__":
    main()
