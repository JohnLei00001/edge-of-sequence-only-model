#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Genos dna_embedding 接口。
从 genos/api.txt 读取端点与密钥(本地, 不入库)。仅用于小批量验证。
"""
import os, re, json, urllib.request, time

ROOT = os.path.join(os.path.dirname(__file__), "..")
API  = os.path.join(ROOT, "genos", "api.txt")

def load_api():
    txt = open(API, encoding="utf-8").read()
    url = re.search(r'"(https://[^"]*dna_embedding[^"]*)"', txt).group(1)
    key = re.search(r"Bearer (sk-\S+)", txt).group(1).strip('"')
    return url, key

def embed(url, key, sequences, model_name="Genos-1.2B", pooling="mean"):
    payload = {"model": "Genos",
               "model_name": model_name,
               "pooling_method": pooling,
               "sequences": sequences}   # 先试 sequences 字段
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key}"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as resp:
        out = json.loads(resp.read().decode())
    return out, time.time() - t0

def main():
    url, key = load_api()
    print("端点:", url)
    print("密钥:", key[:6] + "..." + key[-4:], f"(len {len(key)})")

    test_seqs = [
        "ATCGATAAAAATTTCGAGCTAGC",
        "GGCATAGTCGATCGATCGATCGGGG",
    ]
    for field in ("sequences", "input"):
        print(f"\n=== 尝试字段 '{field}' ===")
        try:
            payload = {"model": "Genos", "model_name": "Genos-1.2B",
                       "pooling_method": "mean", field: test_seqs}
            req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {key}"})
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=120) as resp:
                out = json.loads(resp.read().decode())
            print("耗时 %.1fs" % (time.time() - t0))
            print("响应类型:", type(out).__name__)
            print("响应键:", list(out.keys()) if isinstance(out, dict) else "list")
            # 尝试解析向量
            def find_vecs(obj, depth=0):
                if depth > 5: return None
                if isinstance(obj, dict):
                    for v in obj.values():
                        r = find_vecs(v, depth+1)
                        if r is not None: return r
                elif isinstance(obj, list):
                    if obj and isinstance(obj[0], list) and isinstance(obj[0][0], (int, float)):
                        return obj
                    for v in obj:
                        r = find_vecs(v, depth+1)
                        if r is not None: return r
                return None
            vecs = find_vecs(out)
            if vecs:
                print(f"找到向量: {len(vecs)} 条, 维度={len(vecs[0])}")
                print("前几个维度:", [round(x,4) for x in vecs[0][:5]])
            else:
                print("未找到数值向量, 响应:", json.dumps(out)[:500])
            break
        except Exception as e:
            print("失败:", str(e)[:300])
            print("响应体(若有):", getattr(e, 'read', lambda: b'')()[:300] if hasattr(e,'read') else '')
            # 尝试读取错误响应体
            try:
                print("  err body:", e.read().decode()[:300])
            except Exception:
                pass

if __name__ == "__main__":
    main()
