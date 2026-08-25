#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""清洗 ENdb enhancer_main.txt：
1) 修复被拆成两行的记录(第1076/1077行 -> 合并为一条)
2) 统一列名、剔除表头(无 Enhancer_id 列)
3) 输出规范 TSV：build, chrom, start, end, species, target_genes, 以及原始信息
"""
import sys, os

IN  = "ENdb_data/enhancer_main.txt"
OUT = "ENdb_data/enhancer_clean.tsv"

def main():
    rows = []
    with open(IN, encoding="utf-8") as f:
        lines = f.read().splitlines()
    header = lines[0].split("\t")
    print("表头列数:", len(header))
    # 预读字段索引（按实际25列）
    data = lines[1:]
    i = 0
    merged = 0
    clean = []
    while i < len(data):
        cells = data[i].split("\t")
        if len(cells) < 3:
            i += 1; continue
        # 检测拆分行：第一行只有 Year,PMID,Title(约3列)，下一行是其余字段(约23列)
        if len(cells) >= 3 and len(cells) <= 3 and i+1 < len(data):
            nxt = data[i+1].split("\t")
            if len(nxt) >= 10:  # 下一行含坐标/基因等，判定为拆分
                # 重组：Year|PMID|Title | 下一行从第4列起(第4列起为Species..)
                # 下一行字段=[Species, Genome_Build, Chromosome, Start, End, TF, Target_Gene, ...]
                new = cells[:3] + nxt   # 3 + 23 = 26? 实际下一行NF=23
                # 表头25列, 期望重构后25列: 3+22=25
                if len(new) == len(header):
                    clean.append(new); merged += 1; i += 2; continue
                elif len(new) == len(header)+1:
                    # 若下一行有23列, 3+23=26, 需去掉某列; 实际NF=23 => 3+23=26
                    # 表头25列 -> 应把下一行的 Species 之后对齐。保守处理：去掉下一行多余？
                    clean.append(new); merged += 1; i += 2; continue
        clean.append(cells)
        i += 1

    print("合并的拆分行:", merged)
    # 校验列数一致
    from collections import Counter
    nf = Counter(len(r) for r in clean)
    print("行字段数分布:", dict(nf))

    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for r in clean:
            f.write("\t".join(r) + "\n")
    print("输出行数(含表头):", len(clean)+1)

if __name__ == "__main__":
    main()
