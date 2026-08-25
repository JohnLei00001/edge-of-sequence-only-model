#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验增强子/启动子序列提取结果。"""
import re, sys

def read_fa(path):
    recs = {}
    name = None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith(">"):
            name = line[1:]
            recs[name] = []
        elif name is not None:
            recs[name].append(line)
    return {k: "".join(v) for k, v in recs.items()}

def main():
    enh = read_fa("sequences/enhancers.fa")
    prom = read_fa("sequences/promoters.fa")
    print(f"增强子序列数: {len(enh)}  启动子序列数: {len(prom)}")

    # ---- 增强子长度 ----
    enh_bad = 0
    enh_n = 0
    enh_tot = 0
    for sid, seq in enh.items():
        m = re.search(r":\w+:(\d+)-(\d+)$", sid)
        s, e = int(m.group(1)), int(m.group(2))
        if len(seq) != e - s + 1:
            enh_bad += 1
            if enh_bad <= 5:
                print("  增强子长度错误:", sid, len(seq), e - s + 1)
        enh_n += seq.upper().count("N")
        enh_tot += len(seq)
    print(f"增强子长度错误: {enh_bad}  总N数: {enh_n}/{enh_tot}  "
          f"N比例: {enh_n/enh_tot:.4%}")

    # ---- 启动子长度 ----
    p_bad = 0
    p_full = 0
    p_n = 0
    p_tot = 0
    p_minus = 0
    for sid, seq in prom.items():
        m = re.search(r":\w+:(\d+)-(\d+)\[([+-])\]$", sid)
        s, e, st = int(m.group(1)), int(m.group(2)), m.group(3)
        if len(seq) != e - s + 1:
            p_bad += 1
            if p_bad <= 5:
                print("  启动子长度错误:", sid, len(seq), e - s + 1)
        if len(seq) == 2500:
            p_full += 1
        p_n += seq.upper().count("N")
        p_tot += len(seq)
        if st == "-":
            p_minus += 1
    print(f"启动子长度错误: {p_bad}  完整2500bp: {p_full}/{len(prom)}  "
          f"N比例: {p_n/p_tot:.4%}  负链(已反向互补): {p_minus}")

    # ---- 大写/小写统计 ----
    lower_enh = sum(1 for s in enh.values() for c in s if c.islower())
    lower_prom = sum(1 for s in prom.values() for c in s if c.islower())
    print(f"增强子序列中小写碱基数: {lower_enh}  启动子: {lower_prom}")

if __name__ == "__main__":
    main()
