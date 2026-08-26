#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全量18配置 OOF 收集: 逐样本 out-of-fold 预测 + 逐折明细, 全部落盘留痕。
配置: 9 种 × {GB, RF}。RED=KernelPCA-100(折内fit)。断点续跑。"""
import os, json, time
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT=os.path.join(os.path.dirname(__file__),"..")
CACHE=os.path.join(ROOT,"EP_ATLAS","cache"); OUT=os.path.join(ROOT,"EP_ATLAS","results")
SEED=42
CFGS=['ABC(pure)','ABC+coembed(RED)','ABC+coembed','ABC+embed(RED)','ABC+embed','coembed(RED)','embed(RED)','embed','coembed']
MODELS={'GB':lambda: GradientBoostingClassifier(n_estimators=300,max_depth=4,learning_rate=0.1,random_state=SEED),
        'RF':lambda: RandomForestClassifier(n_estimators=300,max_depth=None,min_samples_leaf=2,random_state=SEED,n_jobs=-1)}

def build_X(cfg,ABC,CO,SEP,tr):
    def kp(M,k=100):
        kp_=KernelPCA(n_components=min(k,M.shape[1]),kernel='rbf',random_state=SEED); kp_.fit(M[tr])
        r=kp_.transform(M); return (r-r.mean(0))/(r.std(0)+1e-9)
    if cfg=='ABC(pure)': return ABC
    if cfg=='ABC+coembed': return np.column_stack([ABC,CO])
    if cfg=='ABC+coembed(RED)': return np.column_stack([ABC,kp(CO)])
    if cfg=='ABC+embed': return np.column_stack([ABC,SEP])
    if cfg=='ABC+embed(RED)': return np.column_stack([ABC,kp(SEP)])
    if cfg=='coembed': return CO
    if cfg=='coembed(RED)': return kp(CO)
    if cfg=='embed': return SEP
    if cfg=='embed(RED)': return kp(SEP)

def main():
    ABC=np.load(os.path.join(CACHE,'ABC.npy')); CO=np.load(os.path.join(CACHE,'CO.npy')); SEP=np.load(os.path.join(CACHE,'SEP.npy'))
    y=np.load(os.path.join(CACHE,'y.npy')); groups=np.load(os.path.join(CACHE,'groups.npy'),allow_pickle=True)
    os.makedirs(OUT,exist_ok=True)
    # 记录
    store=os.path.join(OUT,'oof_all.npz')
    saved={k:np.load(store)[k] for k in np.load(store).files} if os.path.exists(store) else {}
    summary_path=os.path.join(OUT,'oof_all_summary.json')
    summary=json.load(open(summary_path,encoding='utf-8')) if os.path.exists(summary_path) else {}
    fold_info_path=os.path.join(OUT,'oof_all_folds.json')
    fold_info=json.load(open(fold_info_path,encoding='utf-8')) if os.path.exists(fold_info_path) else {}
    for cfg in CFGS:
        for mn,build in MODELS.items():
            key=f'{cfg}|{mn}'
            if key in summary: print(f'跳过已完成: {key}',flush=True); continue
            t0=time.time(); oof=np.zeros(len(y)); folds={}
            for fi,(tr,va) in enumerate(GroupKFold(5).split(ABC,y,groups)):
                X=build_X(cfg,ABC,CO,SEP,tr)
                sc=StandardScaler().fit(X[tr]); clf=build(); clf.fit(sc.transform(X[tr]),y[tr])
                p=clf.predict_proba(sc.transform(X[va]))[:,1]; oof[va]=p
                folds[f'fold{fi}']={'auc':float(roc_auc_score(y[va],p)),'acc':float(accuracy_score(y[va],(p>0.5).astype(int)))}
            saved[key]=oof
            np.savez(store,**saved)  # 即时落盘
            summary[key]={'config':cfg,'model':mn,'oof_auc':float(roc_auc_score(y,oof)),'oof_acc':float(accuracy_score(y,(oof>0.5).astype(int))),'folds':folds}
            json.dump(summary,open(summary_path,'w',encoding='utf-8'),ensure_ascii=False,indent=2)
            json.dump({k:v for k,v in summary.items()},open(fold_info_path,'w',encoding='utf-8'),ensure_ascii=False,indent=2)
            print(f"{key:22s} OOF_AUC={summary[key]['oof_auc']:.4f} Acc={summary[key]['oof_acc']:.4f} [{time.time()-t0:.0f}s]",flush=True)
    # 存 y/groups 一次
    if 'y' not in saved:
        saved['y']=y; saved['groups']=groups; np.savez(store,**saved)
    print(f"\n全部完成: {len(summary)}/18  留痕 → {store} + summary + folds json")

if __name__=='__main__': main()
