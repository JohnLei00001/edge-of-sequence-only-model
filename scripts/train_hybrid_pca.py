#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABC + PCA降维后Embedding 融合分类器 (cis)。
流程: ABC特征(194) 拼接 PCA(embedding 2049→k维) → 分类器, 对比不同 k。
严谨: PCA/StandardScaler 仅 fit 在 train; 分类器 train 拟合, test 评估。
"""
import os, json, time
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.path.join(os.path.dirname(__file__), "..")
SEQ  = os.path.join(ROOT, "EP_ATLAS", "cis", "cis_pairs_sequences.tsv")
EMB  = os.path.join(ROOT, "EP_ATLAS", "embeddings")
OUT  = os.path.join(ROOT, "EP_ATLAS", "models", "hybrid")
SEED = 42
np.random.seed(SEED)
DINUCS=[a+b for a in 'ACGT' for b in 'ACGT']; TRINUCS=[a+b+c for a in 'ACGT' for b in 'ACGT' for c in 'ACGT']

def extract_features(seq):
    seq=str(seq).upper(); n=len(seq)
    if n<3: return np.zeros(93)
    nc={c:seq.count(c) for c in 'ACGT'}; nf=np.array([nc[c]/n for c in 'ACGT'])
    gc=(nc['G']+nc['C'])/n; cpg=seq.count('CG')/n*1000; gpc=seq.count('GC')/n*1000
    ent=-sum((p*np.log2(p)) for p in [nc[c]/n for c in 'ACGT'] if p>0)
    dc=Counter(seq[i:i+2] for i in range(n-1)); td=sum(dc.get(d,0) for d in DINUCS)
    df=np.array([dc.get(d,0)/max(td,1) for d in DINUCS])
    tc=Counter(seq[i:i+3] for i in range(n-2)); tt=sum(tc.get(t,0) for t in TRINUCS)
    tf=np.array([tc.get(t,0)/max(tt,1) for t in TRINUCS])
    tata=sum(1 for i in range(n-4) if seq[i:i+4] in('TATA','AATA')); caat=sum(1 for i in range(n-5) if seq[i:i+5] in('CAATC','CCAAT'))
    hr=0; rl=1
    for i in range(1,n):
        if seq[i]==seq[i-1]: rl+=1
        else:
            if rl>=5: hr+=1
            rl=1
    if rl>=5: hr+=1
    return np.concatenate([nf,[gc],[cpg],[gpc],[ent],df,tf,[tata/n*1000],[caat/n*1000],[hr/n*1000],[np.log1p(n)],[(nc['A']+nc['T'])/max(nc['G']+nc['C'],1)]])

def load_emb(name):
    ids=np.load(os.path.join(EMB,f"{name}_ids.npy"),allow_pickle=True).tolist()
    mat=np.load(os.path.join(EMB,f"{name}_embeddings.npy"),allow_pickle=True)
    return dict(zip(ids,mat))

def main():
    df=pd.read_csv(SEQ,sep='\t')
    enh_e=load_emb("enhancers"); prom_e=load_emb("promoters")
    y=df['label'].values; fold=df['fold'].values
    tr=(fold=='train')|(fold=='val'); te=(fold=='test')
    ytr,yte=y[tr],y[te]
    print(f"cis: n={len(df)} train+val={int(tr.sum())} test={int(te.sum())}")

    # ABC 特征(194) — 与 train_hybrid 一致, calibrator 仅fit train
    enh_mid=(df['enhancer_start']+df['enhancer_end']).values/2.0; prom_mid=(df['promoter_start']+df['promoter_end']).values/2.0
    gd=np.maximum(np.abs(enh_mid-prom_mid),1.0); ch=(gd**(-0.87))*np.exp(-gd/3e6); ch=(ch-ch.min())/(ch.max()-ch.min()+1e-10)
    log10d=np.log10(gd+1)
    enh_f=np.array([extract_features(s) for s in df['enhancer_sequence']]); prom_f=np.array([extract_features(s) for s in df['promoter_sequence']])
    def cos(a,b): return np.array([np.dot(a[i],b[i])/(np.linalg.norm(a[i])*np.linalg.norm(b[i])+1e-10) for i in range(len(a))])
    cos_sim=cos(enh_f,prom_f); cos_tri=cos(enh_f[:,21:85],prom_f[:,21:85])
    gc_sim=1-np.abs(enh_f[:,4]-prom_f[:,4]); cpg_sim=1/(1+np.abs(enh_f[:,5]-prom_f[:,5]))
    euc=np.linalg.norm(enh_f[:,:20]-prom_f[:,:20],axis=1); euc_sim=1/(1+euc)
    cf=np.column_stack([ch,cos_sim,cos_tri,gc_sim,cpg_sim,euc_sim])
    def calib(f):
        c=LogisticRegression(max_iter=1000,C=0.1,random_state=SEED); c.fit(f[tr],y[tr])
        s=c.decision_function(f); return (s-s.min())/(s.max()-s.min()+1e-10)
    act=calib(PCA(10,random_state=SEED).fit_transform(enh_f)); cont=calib(cf)
    abc=np.zeros(len(df))
    for g,idx in df.groupby('gene_id').indices.items():
        num=act[idx]*cont[idx]; denom=num.sum()+1e-10; abc[idx]=num/denom
    ABC=np.column_stack([enh_f,prom_f,cf,log10d[:,None],abc[:,None]])

    # Embedding 特征(2049)
    emb=[]
    for i,row in df.iterrows():
        e=enh_e.get(row['enhancer_id']); p=prom_e.get(row['promoter_id'])
        c=float(np.dot(e,p)/(np.linalg.norm(e)*np.linalg.norm(p)+1e-12))
        emb.append(np.concatenate([e,p,[c]]))
    EMB_all=np.array(emb)

    models={
      'LR':LogisticRegression(max_iter=3000,C=1.0,random_state=SEED),
      'RF':RandomForestClassifier(n_estimators=200,max_depth=12,random_state=SEED,n_jobs=-1),
      'GB':GradientBoostingClassifier(n_estimators=200,max_depth=4,learning_rate=0.1,random_state=SEED),
      'MLP':MLPClassifier(hidden_layer_sizes=(128,64,32),max_iter=500,alpha=0.01,random_state=SEED,early_stopping=True),
    }
    K_LIST=[10,20,50,100]
    print(f"\n{'配置':30s} {'模型':4s} {'AUC':>8s} {'PRAUC':>8s}")

    def run(X, label):
        scaler=StandardScaler().fit(X[tr]); Xtr_s,Xte_s=scaler.transform(X[tr]),scaler.transform(X[te])
        for mname,build in models.items():
            needs = mname in('LR','MLP')
            xtr,xte=(Xtr_s,Xte_s) if needs else (X[tr],X[te])
            clf=build; clf.fit(xtr,ytr); prob=clf.predict_proba(xte)[:,1]
            auc=roc_auc_score(yte,prob); ap=average_precision_score(yte,prob)
            print(f"{label:30s} {mname:4s} {auc:8.4f} {ap:8.4f}")

    print("\n=== ABC-only (基线) ==="); run(ABC,"ABC-only(194)")
    # 各 k 的 PCA 降维融合
    results={}
    for k in K_LIST:
        pca=PCA(n_components=k,random_state=SEED).fit(EMB_all[tr])
        emb_k=pca.transform(EMB_all)
        evr=sum(pca.explained_variance_ratio_)
        print(f"\n=== ABC + PCA-embed({k})  explainedVar={evr:.3f} ===")
        run(np.column_stack([ABC,emb_k]), f"ABC+PCA-{k}")
        results[k]=evr
    with open(os.path.join(OUT,"hybrid_pca_results.json"),"w",encoding="utf-8") as f:
        json.dump({str(k):round(v,4) for k,v in results.items()},f,indent=2)
    print("\n结果已保存 → hybrid_pca_results.json")

if __name__=="__main__": main()
