#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
降维方法 × 维度 网格搜索 (cis)。
目标: 找 ABC + 降维embedding 的最佳融合配置。
方法: PCA / TruncatedSVD / KernelPCA / FastICA / FactorAnalysis / GaussianRandomProjection
维度: 10 / 20 / 50 / 100
评估模型: GradientBoosting (最佳) + LogisticRegression (线性对比)
所有 reducer 仅 fit 在 train; 分类器 train 拟合 test 评估。
"""
import os, json, time
from collections import Counter
import numpy as np, pandas as pd
from sklearn.decomposition import (PCA, TruncatedSVD, KernelPCA, FastICA, FactorAnalysis)
from sklearn.random_projection import GaussianRandomProjection
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=os.path.join(os.path.dirname(__file__),"..")
SEQ=os.path.join(ROOT,"EP_ATLAS","cis","cis_pairs_sequences.tsv")
EMB=os.path.join(ROOT,"EP_ATLAS","embeddings")
OUT=os.path.join(ROOT,"EP_ATLAS","models","hybrid")
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

def main():
    df=pd.read_csv(SEQ,sep='\t')
    enh_e=load_emb("enhancers"); prom_e=load_emb("promoters")
    y=df['label'].values; fold=df['fold'].values
    tr=(fold=='train')|(fold=='val'); te=(fold=='test')
    ytr,yte=y[tr],y[te]
    ABC=build_abc(df,tr,y)
    emb=[]
    for i,row in df.iterrows():
        e=enh_e.get(row['enhancer_id']); p=prom_e.get(row['promoter_id'])
        c=float(np.dot(e,p)/(np.linalg.norm(e)*np.linalg.norm(p)+1e-12))
        emb.append(np.concatenate([e,p,[c]]))
    EMB_all=np.array(emb)

    methods={
      'PCA': lambda k: PCA(n_components=k,random_state=SEED),
      'TruncatedSVD': lambda k: TruncatedSVD(n_components=k,random_state=SEED),
      'KernelPCA': lambda k: KernelPCA(n_components=k,kernel='rbf',random_state=SEED),
      'FastICA': lambda k: FastICA(n_components=k,random_state=SEED,max_iter=500),
      'FactorAnalysis': lambda k: FactorAnalysis(n_components=k,random_state=SEED),
      'GaussianRP': lambda k: GaussianRandomProjection(n_components=k,random_state=SEED),
    }
    K=[10,20,50,100]
    models={
      'GB': GradientBoostingClassifier(n_estimators=200,max_depth=4,learning_rate=0.1,random_state=SEED),
      'LR': LogisticRegression(max_iter=3000,C=1.0,random_state=SEED),
    }
    print(f"{'方法':16s} {'k':>5s} {'模型':3s} {'AUC':>8s} {'PRAUC':>8s} {'耗时':>6s}")
    allres={}
    scaler_abc=StandardScaler().fit(ABC[tr]); ABCtr,ABCte=scaler_abc.transform(ABC[tr]),scaler_abc.transform(ABC[te])
    for mname,mk in methods.items():
        for k in K:
            t0=time.time()
            try:
                red=mk(k); red.fit(EMB_all[tr]); embk=red.transform(EMB_all)
                embk=(embk-embk.mean(0))/(embk.std(0)+1e-9)
                X=np.column_stack([ABC,embk])
                scaler=StandardScaler().fit(X[tr]); Xtr_s,Xte_s=scaler.transform(X[tr]),scaler.transform(X[te])
                for mn,build in models.items():
                    needs = (mn=='LR')
                    xtr,xte=(Xtr_s,Xte_s) if needs else (X[tr],X[te])
                    clf=build; clf.fit(xtr,ytr); prob=clf.predict_proba(xte)[:,1]
                    auc=roc_auc_score(yte,prob); ap=average_precision_score(yte,prob)
                    allres.setdefault(f"{mname}_k{k}",{})[mn]={'auc':round(float(auc),4),'prauc':round(float(ap),4)}
                    print(f"{mname:16s} {k:5d} {mn:3s} {auc:8.4f} {ap:8.4f} {time.time()-t0:6.1f}s")
            except Exception as e:
                print(f"{mname:16s} {k:5d}  ERR {str(e)[:80]}")
    with open(os.path.join(OUT,"dr_grid_results.json"),"w",encoding="utf-8") as f:
        json.dump(allres,f,indent=2,ensure_ascii=False)
    print("\n结果 → dr_grid_results.json")

if __name__=="__main__": main()
