#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABC_Score + Embedding 融合分类器 (cis)。
对比三种特征: ABC-only(194) / Embedding-only(2049) / Hybrid(2243)
严谨: 内部 calibrator (activity/contact) 仅 fit 在 train, 避免泄漏;
      外层分类器在 train 拟合, test 评估; 特征标准化仅 fit train。
模型: LR / RF / GB / MLP
"""
import os, csv, sys, time, json
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, confusion_matrix

ROOT = os.path.join(os.path.dirname(__file__), "..")
SEQ  = os.path.join(ROOT, "EP_ATLAS", "cis", "cis_pairs_sequences.tsv")
LAB  = os.path.join(ROOT, "EP_ATLAS", "cis", "cis_pairs_labeled.tsv")
EMB  = os.path.join(ROOT, "EP_ATLAS", "embeddings")
OUT  = os.path.join(ROOT, "EP_ATLAS", "models", "hybrid")
SEED = 42
np.random.seed(SEED)

DINUCS = [a+b for a in 'ACGT' for b in 'ACGT']
TRINUCS = [a+b+c for a in 'ACGT' for b in 'ACGT' for c in 'ACGT']

def extract_features(seq):
    seq = str(seq).upper(); n = len(seq)
    if n < 3: return np.zeros(93)
    nc = {c: seq.count(c) for c in 'ACGT'}
    nf = np.array([nc[c]/n for c in 'ACGT'])
    gc = (nc['G']+nc['C'])/n; cpg = seq.count('CG')/n*1000; gpc = seq.count('GC')/n*1000
    ent = -sum((p*np.log2(p)) for p in [nc[c]/n for c in 'ACGT'] if p>0)
    dc = Counter(seq[i:i+2] for i in range(n-1)); td = sum(dc.get(d,0) for d in DINUCS)
    df = np.array([dc.get(d,0)/max(td,1) for d in DINUCS])
    tc = Counter(seq[i:i+3] for i in range(n-2)); tt = sum(tc.get(t,0) for t in TRINUCS)
    tf = np.array([tc.get(t,0)/max(tt,1) for t in TRINUCS])
    tata = sum(1 for i in range(n-4) if seq[i:i+4] in ('TATA','AATA'))
    caat = sum(1 for i in range(n-5) if seq[i:i+5] in ('CAATC','CCAAT'))
    hr = 0; rl = 1
    for i in range(1,n):
        if seq[i]==seq[i-1]: rl+=1
        else:
            if rl>=5: hr+=1
            rl=1
    if rl>=5: hr+=1
    return np.concatenate([nf,[gc],[cpg],[gpc],[ent],df,tf,
                           [tata/n*1000],[caat/n*1000],[hr/n*1000],[np.log1p(n)],
                           [(nc['A']+nc['T'])/max(nc['G']+nc['C'],1)]])

def load_emb(name):
    ids=np.load(os.path.join(EMB,f"{name}_ids.npy"),allow_pickle=True).tolist()
    mat=np.load(os.path.join(EMB,f"{name}_embeddings.npy"),allow_pickle=True)
    return dict(zip(ids,mat))

def main():
    df = pd.read_csv(SEQ, sep='\t')
    # 与 embedding 特征对齐(按 enhancer_id+promoter_id)
    enh_e = load_emb("enhancers"); prom_e = load_emb("promoters")
    y = df['label'].values
    fold = df['fold'].values
    print(f"cis 样本: {len(df)} (pos={int((y==1).sum())} neg={int((y==0).sum())})")
    train_m = (fold=='train')|(fold=='val'); test_m = (fold=='test')
    print(f"train+val={int(train_m.sum())}  test={int(test_m.sum())}")

    # ---- ABC 特征 ----
    enh_mid=(df['enhancer_start']+df['enhancer_end']).values/2.0
    prom_mid=(df['promoter_start']+df['promoter_end']).values/2.0
    gd=np.maximum(np.abs(enh_mid-prom_mid),1.0).astype(float)
    GAMMA,D_DECAY=0.87,3e6
    ch=(gd**(-GAMMA))*np.exp(-gd/D_DECAY)
    ch=(ch-ch.min())/(ch.max()-ch.min()+1e-10)
    log10d=np.log10(gd+1)
    print("[1] 计算 93维序列特征...")
    enh_f=np.array([extract_features(s) for s in df['enhancer_sequence']])
    prom_f=np.array([extract_features(s) for s in df['promoter_sequence']])
    def cos(a,b): return np.array([np.dot(a[i],b[i])/(np.linalg.norm(a[i])*np.linalg.norm(b[i])+1e-10) for i in range(len(a))])
    cos_sim=cos(enh_f,prom_f)
    cos_tri=cos(enh_f[:,21:85],prom_f[:,21:85])
    gc_sim=1-np.abs(enh_f[:,4]-prom_f[:,4])
    cpg_sim=1/(1+np.abs(enh_f[:,5]-prom_f[:,5]))
    euc=np.linalg.norm(enh_f[:,:20]-prom_f[:,:20],axis=1); euc_sim=1/(1+euc)
    contact_feats=np.column_stack([ch,cos_sim,cos_tri,gc_sim,cpg_sim,euc_sim])

    # 仅 train 拟合 calibrator(防泄漏)
    def calib_score(feats):
        clf=LogisticRegression(max_iter=1000,C=0.1,random_state=SEED)
        clf.fit(feats[train_m],y[train_m])
        s=clf.decision_function(feats); s=(s-s.min())/(s.max()-s.min()+1e-10)
        return s
    activity_score=calib_score(PCA(10,random_state=SEED).fit_transform(enh_f))
    contact_score=calib_score(contact_feats)
    # ABC = activity*contact / per-gene
    abc=np.zeros(len(df))
    for g,idx in df.groupby('gene_id').indices.items():
        num=activity_score[idx]*contact_score[idx]; denom=num.sum()+1e-10; abc[idx]=num/denom
    abc_features=np.column_stack([enh_f,prom_f,contact_feats,log10d[:,None],abc[:,None]])
    print(f"[2] ABC 特征维度: {abc_features.shape[1]}")

    # ---- Embedding 特征 ----
    emb_feats=[]
    miss=0
    for i,row in df.iterrows():
        e=enh_e.get(row['enhancer_id']); p=prom_e.get(row['promoter_id'])
        if e is None or p is None: miss+=1; emb_feats.append(np.zeros(2049)); continue
        c=float(np.dot(e,p)/(np.linalg.norm(e)*np.linalg.norm(p)+1e-12))
        emb_feats.append(np.concatenate([e,p,[c]]))
    emb_feats=np.array(emb_feats); print(f"[3] Embedding 特征维度: {emb_feats.shape[1]} (缺失{miss})")

    feats = {'ABC':abc_features, 'Embed':emb_feats, 'Hybrid':np.column_stack([abc_features,emb_feats])}
    models = {
        'LR': LogisticRegression(max_iter=3000,C=1.0,random_state=SEED),
        'RF': RandomForestClassifier(n_estimators=200,max_depth=12,random_state=SEED,n_jobs=-1),
        'GB': GradientBoostingClassifier(n_estimators=200,max_depth=4,learning_rate=0.1,random_state=SEED),
        'MLP': MLPClassifier(hidden_layer_sizes=(128,64,32),max_iter=500,alpha=0.01,random_state=SEED,early_stopping=True),
    }
    print("\n[4] 训练与评估 (test set):")
    print(f"{'特征':6s} {'模型':5s} {'AUC':>8s} {'PRAUC':>8s} {'Acc':>7s} {'F1':>7s} {'trainAUC':>9s}")
    results={}
    for fname,X in feats.items():
        scaler=StandardScaler().fit(X[train_m])
        Xtr,Xte=X[train_m],X[test_m]; Xtr_s,Xte_s=scaler.transform(Xtr),scaler.transform(Xte)
        ytr,yte=y[train_m],y[test_m]
        for mname,build in models.items():
            needs_scale = mname in ('LR','MLP')
            tr,te = (Xtr_s,Xte_s) if needs_scale else (Xtr,Xte)
            clf=build; clf.fit(tr,ytr)
            prob=clf.predict_proba(te)[:,1]
            auc=roc_auc_score(yte,prob); ap=average_precision_score(yte,prob)
            acc=accuracy_score(yte,(prob>0.5).astype(int)); f1=f1_score(yte,(prob>0.5).astype(int))
            tauc=roc_auc_score(ytr,clf.predict_proba(tr)[:,1])
            results[(fname,mname)]={'auc':auc,'ap':ap,'acc':acc,'f1':f1,'tauc':tauc,'clf':clf,'scaler':scaler}
            print(f"{fname:6s} {mname:5s} {auc:8.4f} {ap:8.4f} {acc:7.4f} {f1:7.4f} {tauc:9.4f}")

    # 保存结果
    os.makedirs(OUT,exist_ok=True)
    sumr={}
    for k,v in results.items(): sumr[f"{k[0]}_{k[1]}"]={kk:round(float(vv),4) for kk,vv in v.items() if kk in('auc','ap','acc','f1','tauc')}
    with open(os.path.join(OUT,"hybrid_results.json"),"w",encoding="utf-8") as f: json.dump(sumr,f,indent=2)
    print("\n结果已保存 →",os.path.join(OUT,"hybrid_results.json"))

if __name__=="__main__":
    main()
