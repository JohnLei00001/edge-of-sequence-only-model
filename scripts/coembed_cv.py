#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
co-embedding 评估 (cis): 用 GB + 按增强子分组CV, 多档降维数。
对比: ABC-only 基线 / ABC+co-embed(降维,k) / co-embed(降维,k) 单独
每个 k: 分组5折CV, 每折内 KernelPCA+scaler 仅fit train, GB 训练, 报告 AUC mean±std。
"""
import os, json
from collections import Counter
import numpy as np, pandas as pd
from sklearn.decomposition import (PCA, TruncatedSVD, KernelPCA, FastICA, FactorAnalysis)
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=os.path.join(os.path.dirname(__file__),"..")
SEQ=os.path.join(ROOT,"EP_ATLAS","cis","cis_pairs_sequences.tsv")
COEMB=os.path.join(ROOT,"EP_ATLAS","embeddings","co_cis_embeddings.npy")
COIDS=os.path.join(ROOT,"EP_ATLAS","embeddings","co_cis_ids.npy")
OUT=os.path.join(ROOT,"EP_ATLAS","report")
SEED=42; np.random.seed(SEED)
DINUCS=[a+b for a in 'ACGT' for b in 'ACGT']; TRINUCS=[a+b+c for a in 'ACGT' for b in 'ACGT' for c in 'ACGT']
K_DIMS=[10,20,30,50,100,200]
METHODS={
  'PCA':lambda k: PCA(n_components=k,random_state=SEED),
  'TruncatedSVD':lambda k: TruncatedSVD(n_components=k,random_state=SEED),
  'KernelPCA':lambda k: KernelPCA(n_components=k,kernel='rbf',random_state=SEED),
  'FastICA':lambda k: FastICA(n_components=k,random_state=SEED,max_iter=400),
  'FactorAnalysis':lambda k: FactorAnalysis(n_components=k,random_state=SEED),
  'GaussianRP':lambda k: GaussianRandomProjection(n_components=k,random_state=SEED),
}

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

def build_abc(df,tr,y):
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
    return np.column_stack([enh_f,prom_f,cf,log10d[:,None],abc[:,None]])

def cv_gb(X, y, groups, n_splits=5):
    aucs,aps=[],[]
    gkf=GroupKFold(n_splits=n_splits)
    for tr,va in gkf.split(X,y,groups):
        ytr,yva=y[tr],y[va]
        scaler=StandardScaler().fit(X[tr]); Xtr,Xva=scaler.transform(X[tr]),scaler.transform(X[va])
        clf=GradientBoostingClassifier(n_estimators=300,max_depth=4,learning_rate=0.1,random_state=SEED)
        clf.fit(Xtr,ytr); prob=clf.predict_proba(Xva)[:,1]
        aucs.append(roc_auc_score(yva,prob)); aps.append(average_precision_score(yva,prob))
    return np.mean(aucs),np.std(aucs),np.mean(aps),np.std(aps)

def main():
    df=pd.read_csv(SEQ,sep='\t'); y=df['label'].values; groups=df['enhancer_id'].values
    tr_all=np.arange(len(df))
    ABC=build_abc(df,tr_all,y)
    # co-embedding
    coids=np.load(COIDS,allow_pickle=True).tolist()
    coemb=np.load(COEMB,allow_pickle=True)
    co_map=dict(zip(coids,coemb))
    keys=[f"{e}__{p}" for e,p in zip(df['enhancer_id'],df['promoter_id'])]
    CO=np.array([co_map[k] for k in keys])
    print(f"co-embedding 矩阵: {CO.shape}  缺失: {sum(1 for k in keys if k not in co_map)}")

    print(f"\n{'配置':40s} {'AUC':>12s} {'PR-AUC':>12s}")
    res={}
    # ABC 基线
    m,s,mp,sp=cv_gb(ABC,y,groups); res['ABC(194)']={'auc_m':m,'auc_s':s,'ap_m':mp,'ap_s':sp}
    print(f"{'ABC (基线)':40s} {m:.4f}±{s:.4f}  {mp:.4f}±{sp:.4f}")
    for cfg,base in [('ABC+coembed',ABC),('coembed-only',None)]:
        for mname,mk in METHODS.items():
            for k in K_DIMS:
                aucs,aps=[],[]
                for tr,va in GroupKFold(n_splits=5).split(CO,y,groups):
                    kp=mk(k); kp.fit(CO[tr]); red=kp.transform(CO); red=(red-red.mean(0))/(red.std(0)+1e-9)
                    X = np.column_stack([ABC,red]) if base is not None else red
                    scaler=StandardScaler().fit(X[tr]); Xtr,Xva=scaler.transform(X[tr]),scaler.transform(X[va])
                    clf=GradientBoostingClassifier(n_estimators=300,max_depth=4,learning_rate=0.1,random_state=SEED)
                    clf.fit(Xtr,y[tr]); prob=clf.predict_proba(Xva)[:,1]
                    aucs.append(roc_auc_score(y[va],prob)); aps.append(average_precision_score(y[va],prob))
                label=f'{cfg}({mname},{k})'
                res[label]={'auc_m':float(np.mean(aucs)),'auc_s':float(np.std(aucs)),
                            'ap_m':float(np.mean(aps)),'ap_s':float(np.std(aps))}
                print(f"{label:40s} {np.mean(aucs):.4f}±{np.std(aucs):.4f}  {np.mean(aps):.4f}±{np.std(aps):.4f}",flush=True)
    with open(os.path.join(OUT,"cn_coembed_cv.json"),"w",encoding="utf-8") as f:
        json.dump(res,f,indent=2,ensure_ascii=False)
    print("\n结果 → cn_coembed_cv.json")

if __name__=='__main__': main()
