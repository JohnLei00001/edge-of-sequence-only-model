#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""8配置 × {RF, GB} 分组5折CV对比, RED=KernelPCA-100(折内fit,无泄漏)。逐条落盘留痕。
配置: ABC+coembed(RED)/ABC+coembed/ABC+embed(RED)/ABC+embed/coembed(RED)/embed(RED)/embed/coembed"""
import os, json, time
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

ROOT=os.path.join(os.path.dirname(__file__),"..")
CACHE=os.path.join(ROOT,"EP_ATLAS","cache"); OUT=os.path.join(ROOT,"EP_ATLAS","results")
SEED=42
RED_K=100
CFG_ORDER=['ABC+coembed(RED)','ABC+coembed','ABC+embed(RED)','ABC+embed',
           'coembed(RED)','embed(RED)','embed','coembed']
MODELS={'GB':lambda: GradientBoostingClassifier(n_estimators=300,max_depth=4,learning_rate=0.1,random_state=SEED),
        'RF':lambda: RandomForestClassifier(n_estimators=300,max_depth=None,min_samples_leaf=2,random_state=SEED,n_jobs=-1)}

def build_X(cfg,ABC,CO,SEP,tr):
    """按配置构建特征(降维在tr内fit)。返回全量X。"""
    def kp100(M):
        kp=KernelPCA(n_components=min(RED_K,M.shape[1]),kernel='rbf',random_state=SEED); kp.fit(M[tr])
        r=kp.transform(M); return (r-r.mean(0))/(r.std(0)+1e-9)
    if cfg=='ABC+coembed(RED)': return np.column_stack([ABC,kp100(CO)])
    if cfg=='ABC+coembed':      return np.column_stack([ABC,CO])
    if cfg=='ABC+embed(RED)':   return np.column_stack([ABC,kp100(SEP)])
    if cfg=='ABC+embed':        return np.column_stack([ABC,SEP])
    if cfg=='coembed(RED)':     return kp100(CO)
    if cfg=='embed(RED)':       return kp100(SEP)
    if cfg=='embed':            return SEP
    if cfg=='coembed':          return CO

def main():
    ABC=np.load(os.path.join(CACHE,'ABC.npy')); CO=np.load(os.path.join(CACHE,'CO.npy')); SEP=np.load(os.path.join(CACHE,'SEP.npy'))
    y=np.load(os.path.join(CACHE,'y.npy')); groups=np.load(os.path.join(CACHE,'groups.npy'),allow_pickle=True)
    os.makedirs(OUT,exist_ok=True); path=os.path.join(OUT,'compare_8configs.json')
    results=json.load(open(path,encoding='utf-8')) if os.path.exists(path) else {}
    print(f"{'配置':20s} {'模型':3s} {'AUC':>8s} {'PRAUC':>8s} {'Acc':>7s} {'F1':>7s} {'耗时':>6s}",flush=True)
    for cfg in CFG_ORDER:
        X=build_X(cfg,ABC,CO,SEP,np.arange(len(y)))
        for mn,build in MODELS.items():
            key=f"{cfg}|{mn}"
            if key in results: continue
            t0=time.time(); aucs,aps,accs,f1s=[],[],[],[]
            for tr,va in GroupKFold(5).split(X,y,groups):
                sc=StandardScaler().fit(X[tr]); clf=build(); clf.fit(sc.transform(X[tr]),y[tr])
                p=clf.predict_proba(sc.transform(X[va]))[:,1]; pred=(p>0.5).astype(int)
                aucs.append(roc_auc_score(y[va],p)); aps.append(average_precision_score(y[va],p))
                accs.append(accuracy_score(y[va],pred)); f1s.append(f1_score(y[va],pred))
            results[key]={'config':cfg,'model':mn,'auc_m':float(np.mean(aucs)),'auc_s':float(np.std(aucs)),
                          'ap_m':float(np.mean(aps)),'acc_m':float(np.mean(accs)),'f1_m':float(np.mean(f1s))}
            json.dump(results,open(path,'w',encoding='utf-8'),ensure_ascii=False,indent=2)  # 逐条落盘
            print(f"{cfg:20s} {mn:3s} {np.mean(aucs):.4f} {np.mean(aps):.4f} {np.mean(accs):.4f} {np.mean(f1s):.4f} {time.time()-t0:5.0f}s",flush=True)
    # 汇总: 各配置最佳模型
    print("\n=== 各配置最佳(按AUC) ===")
    for cfg in CFG_ORDER:
        cand={k:v for k,v in results.items() if v['config']==cfg}
        b=max(cand.values(),key=lambda v:v['auc_m'])
        print(f"  {cfg:20s} {b['model']} AUC={b['auc_m']:.4f} PRAUC={b['ap_m']:.4f} Acc={b['acc_m']:.4f}")
    print("\n→ compare_8configs.json (留痕)")
if __name__=='__main__': main()
