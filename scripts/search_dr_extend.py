#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DR搜索-扩展: 向上搜更高维(400..1024)确认峰值, 逐条落盘。"""
import os, json, time, sys
import numpy as np
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=os.path.join(os.path.dirname(__file__),"..")
CACHE=os.path.join(ROOT,"EP_ATLAS","cache"); OUT=os.path.join(ROOT,"EP_ATLAS","results")
SEED=42
METHODS={
  'PCA':lambda k: PCA(n_components=k,random_state=SEED),
  'KernelPCA':lambda k: KernelPCA(n_components=k,kernel='rbf',random_state=SEED),
  'TruncatedSVD':lambda k: TruncatedSVD(n_components=k,random_state=SEED),
}
DIMS=[400,500,600,700,800,900,1000,1024]

def cv_gb(ABC,M,y,groups,mk,k):
    aucs,aps=[],[]
    for tr,va in GroupKFold(5).split(ABC,y,groups):
        red_=mk(k); red_.fit(M[tr]); red=red_.transform(M)
        red=(red-red.mean(0))/(red.std(0)+1e-9)
        X=np.column_stack([ABC,red])
        sc=StandardScaler().fit(X[tr]); clf=GradientBoostingClassifier(n_estimators=300,max_depth=4,learning_rate=0.1,random_state=SEED)
        clf.fit(sc.transform(X[tr]),y[tr]); p=clf.predict_proba(sc.transform(X[va]))[:,1]
        aucs.append(roc_auc_score(y[va],p)); aps.append(average_precision_score(y[va],p))
    return np.mean(aucs),np.std(aucs),np.mean(aps)

def main():
    ABC=np.load(os.path.join(CACHE,'ABC.npy')); CO=np.load(os.path.join(CACHE,'CO.npy'))
    y=np.load(os.path.join(CACHE,'y.npy')); groups=np.load(os.path.join(CACHE,'groups.npy'),allow_pickle=True)
    # 已有结果
    path=os.path.join(OUT,'dr_search.json')
    results=json.load(open(path,encoding='utf-8')) if os.path.exists(path) else {}
    print(f"已加载 {len(results)} 条已有结果",flush=True)
    print(f"{'方法':12s} {'k':>4s} {'AUC':>10s} {'PR-AUC':>10s} {'耗时':>6s}",flush=True)
    for mname,mk in METHODS.items():
        for k in DIMS:
            key=f"{mname}_k{k}"
            if key in results: continue
            t0=time.time()
            try:
                m,s,mp=cv_gb(ABC,CO,y,groups,mk,k)
            except Exception as e:
                print(f"{mname:12s} {k:4d} ERR {str(e)[:60]}",flush=True); continue
            results[key]={'method':mname,'k':k,'auc_m':float(m),'auc_s':float(s),'ap_m':float(mp)}
            json.dump(results,open(path,'w',encoding='utf-8'),ensure_ascii=False,indent=2)  # 逐条落盘
            print(f"{mname:12s} {k:4d} {m:.4f}±{s:.4f} {mp:.4f} {time.time()-t0:5.0f}s",flush=True)
    # 选最佳 + 边界检查
    best=max(results,key=lambda kk:results[kk]['auc_m'])
    bb=results[best]
    allk=sorted(set(r['k'] for r in results.values() if r['method']==bb['method']))
    kk=bb['k']
    interior = kk!=allk[0] and kk!=allk[-1]
    print(f"\n[RESULT] 全局最佳: {best}  AUC={bb['auc_m']:.4f} PR={bb['ap_m']:.4f}",flush=True)
    print(f"[RESULT] 该方法取样维数: {allk}",flush=True)
    print(f"[RESULT] 峰值维数 {kk} 在内部={interior}",flush=True)
    json.dump({'best':best,'best_k':kk,'interior':interior,'n_configs':len(results)},
              open(os.path.join(OUT,'dr_search_best.json'),'w'),ensure_ascii=False,indent=2)
    print("→ dr_search.json 已更新(留痕)",flush=True)
if __name__=='__main__': main()
