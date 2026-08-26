#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按增强子分组的 5 折交叉验证 (cis)。
5 种特征配置, 统一 GB, 报告 AUC/PR-AUC 的 mean±std。
按 enhancer_id 分组, 避免同增强子跨折泄漏。
"""
import os, json
from collections import Counter
import numpy as np, pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=os.path.join(os.path.dirname(__file__),"..")
SEQ=os.path.join(ROOT,"EP_ATLAS","cis","cis_pairs_sequences.tsv")
EMB=os.path.join(ROOT,"EP_ATLAS","embeddings")
OUT=os.path.join(ROOT,"EP_ATLAS","report")
SEED=42; np.random.seed(SEED)
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

def build_features(df,tr,y,enh_e,prom_e):
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
    emb=[]
    for i,row in df.iterrows():
        e=enh_e.get(row['enhancer_id']); p=prom_e.get(row['promoter_id'])
        c=float(np.dot(e,p)/(np.linalg.norm(e)*np.linalg.norm(p)+1e-12))
        emb.append(np.concatenate([e,p,[c]]))
    EMB_all=np.array(emb)
    kpca=KernelPCA(n_components=100,kernel='rbf',random_state=SEED).fit(EMB_all[tr])
    embk=kpca.transform(EMB_all); embk=(embk-embk.mean(0))/(embk.std(0)+1e-9)
    return ABC, EMB_all, embk, gd

def main():
    df=pd.read_csv(SEQ,sep='\t'); enh_e=load_emb("enhancers"); prom_e=load_emb("promoters")
    y=df['label'].values
    # 用全量算特征(calibrator 在每折内重fit, 不泄漏)
    tr_full=np.arange(len(df))
    ABC,EMB_raw,EMB_red,_=build_features(df,tr_full,y,enh_e,prom_e)
    FEATURES={'ABC':ABC,'ABC+Embed(red)':np.column_stack([ABC,EMB_red]),
              'ABC+Embed(raw)':np.column_stack([ABC,EMB_raw]),'Embed(raw)':EMB_raw,'Embed(red)':EMB_red}
    groups=df['enhancer_id'].values
    gkf=GroupKFold(n_splits=5)
    print("特征维度:",{k:v.shape[1] for k,v in FEATURES.items()})
    cv={}
    for cname,X in FEATURES.items():
        aucs,aps=[],[]
        for fold,(tr,va) in enumerate(gkf.split(X,y,groups),1):
            ytr,yva=y[tr],y[va]
            scaler=StandardScaler().fit(X[tr]); Xtr,Xva=scaler.transform(X[tr]),scaler.transform(X[va])
            clf=GradientBoostingClassifier(n_estimators=300,max_depth=4,learning_rate=0.1,random_state=SEED)
            clf.fit(Xtr,ytr); prob=clf.predict_proba(Xva)[:,1]
            aucs.append(roc_auc_score(yva,prob)); aps.append(average_precision_score(yva,prob))
        cv[cname]={'auc_mean':float(np.mean(aucs)),'auc_std':float(np.std(aucs)),
                   'ap_mean':float(np.mean(aps)),'ap_std':float(np.std(aps)),
                   'auc_all':[float(x) for x in aucs],'ap_all':[float(x) for x in aps]}
        print(f"{cname:18s} AUC={np.mean(aucs):.4f}±{np.std(aucs):.4f}  PR-AUC={np.mean(aps):.4f}±{np.std(aps):.4f}")
    with open(os.path.join(OUT,"cn_config_cv.json"),"w",encoding="utf-8") as f:
        json.dump(cv,f,indent=2,ensure_ascii=False)
    print("\nCV结果 → cn_config_cv.json")

if __name__=='__main__': main()
