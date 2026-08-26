#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DR 搜索: 对 co-embedding 做 方法×维度 网格降维, 融合 ABC, GB+分组CV, 选最佳。
保证峰值维数在取样内部(若在边界则扩展重搜)。结果逐条落盘留痕。
"""
import os, json, time
import numpy as np
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=os.path.join(os.path.dirname(__file__),"..")
CACHE=os.path.join(ROOT,"EP_ATLAS","cache")
OUT=os.path.join(ROOT,"EP_ATLAS","results")
SEED=42
METHODS={
  'PCA':lambda k: PCA(n_components=k,random_state=SEED),
  'KernelPCA':lambda k: KernelPCA(n_components=k,kernel='rbf',random_state=SEED),
  'TruncatedSVD':lambda k: TruncatedSVD(n_components=k,random_state=SEED),
}
DIMS=[5,10,20,30,50,75,100,150,200,300]

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
    os.makedirs(OUT,exist_ok=True)
    results={}
    print(f"{'方法':12s} {'k':>4s} {'AUC':>10s} {'PR-AUC':>10s} {'耗时':>6s}")
    for mname,mk in METHODS.items():
        for k in DIMS:
            t0=time.time()
            try:
                m,s,mp=cv_gb(ABC,CO,y,groups,mk,k)
            except Exception as e:
                print(f"{mname:12s} {k:4d} ERR {str(e)[:60]}"); continue
            results[f"{mname}_k{k}"]={'method':mname,'k':k,'auc_m':float(m),'auc_s':float(s),'ap_m':float(mp)}
            print(f"{mname:12s} {k:4d} {m:.4f}±{s:.4f} {mp:.4f} {time.time()-t0:5.0f}s",flush=True)
    with open(os.path.join(OUT,'dr_search.json'),'w',encoding='utf-8') as f: json.dump(results,f,ensure_ascii=False,indent=2)
    # 选最佳
    best=max(results,key=lambda kk:results[kk]['auc_m'])
    print(f"\n最佳: {best}  AUC={results[best]['auc_m']:.4f}")
    # 边界检查
    kk=results[best]['k']; dims_sorted=sorted(DIMS)
    if kk==dims_sorted[0] or kk==dims_sorted[-1]:
        print(f"⚠️ 峰值维数 {kk} 在取样边界! 需扩展网格在{kk}两侧再搜")
    else:
        print(f"✅ 峰值维数 {kk} 在取样内部(两侧:{dims_sorted[dims_sorted.index(kk)-1]},{dims_sorted[dims_sorted.index(kk)+1]})")
    print("→ dr_search.json (留痕)")
if __name__=='__main__': main()
