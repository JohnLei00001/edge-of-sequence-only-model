#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
有监督降维 (PLS / 带正则LDA) 在折内 fit, 对比无监督 KPCA。
配置: ABC+coembed 分别用 无监督KPCA100 / PLS(k) / LDA(shrinkage) 降维, GB+分组CV。
"""
import os, json
from collections import Counter
import numpy as np, pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=os.path.join(os.path.dirname(__file__),"..")
SEQ=os.path.join(ROOT,"EP_ATLAS","cis","cis_pairs_sequences.tsv")
EMB=os.path.join(ROOT,"EP_ATLAS","embeddings"); OUT=os.path.join(ROOT,"EP_ATLAS","report")
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
        c=LogisticRegression(max_iter=1000,C=0.1,random_state=SEED); c.fit(f[tr],y[tr]); s=c.decision_function(f)
        return (s-s.min())/(s.max()-s.min()+1e-10)
    act=calib(PCA(10,random_state=SEED).fit_transform(enh_f)); cont=calib(cf)
    abc=np.zeros(len(df))
    for g,idx in df.groupby('gene_id').indices.items():
        num=act[idx]*cont[idx]; denom=num.sum()+1e-10; abc[idx]=num/denom
    return np.column_stack([enh_f,prom_f,cf,log10d[:,None],abc[:,None]])

def cv_gb(ABC, CO, y, groups, reducer):
    """reducer(CO_tr, y_tr) -> transform(CO) 返回降维后的全量矩阵"""
    aucs,aps=[],[]
    for tr,va in GroupKFold(5).split(ABC,y,groups):
        if reducer is None:
            X=ABC
        else:
            red=reducer(CO[tr],y[tr],CO)
            red=(red-red.mean(0))/(red.std(0)+1e-9)
            X=np.column_stack([ABC,red])
        sc=StandardScaler().fit(X[tr]); Xtr,Xva=sc.transform(X[tr]),sc.transform(X[va])
        clf=GradientBoostingClassifier(n_estimators=300,max_depth=4,learning_rate=0.1,random_state=SEED)
        clf.fit(Xtr,y[tr]); p=clf.predict_proba(Xva)[:,1]
        aucs.append(roc_auc_score(y[va],p)); aps.append(average_precision_score(y[va],p))
    return np.mean(aucs),np.std(aucs),np.mean(aps)

def mk_kpca(k=100):
    def f(CoTr,ytr,Co):
        kp=KernelPCA(n_components=k,kernel='rbf',random_state=SEED).fit(CoTr); return kp.transform(Co)
    return f
def mk_pls(k):
    def f(CoTr,ytr,Co):
        sc=StandardScaler().fit(CoTr); CoTrs=sc.transform(CoTr)
        pls=PLSRegression(n_components=k); pls.fit(CoTrs,ytr)
        return pls.transform(sc.transform(Co))
    return f
def mk_lda():
    def f(CoTr,ytr,Co):
        sc=StandardScaler().fit(CoTr)
        lda=LinearDiscriminantAnalysis(n_components=1,solver='eigen',shrinkage='auto')
        lda.fit(sc.transform(CoTr),ytr)
        return lda.transform(sc.transform(Co))
    return f

def main():
    df=pd.read_csv(SEQ,sep='\t'); y=df['label'].values; groups=df['enhancer_id'].values
    tr_all=np.arange(len(df)); ABC=build_abc(df,tr_all,y)
    co_map=dict(zip(np.load(os.path.join(EMB,'co_cis_ids.npy'),allow_pickle=True).tolist(),np.load(os.path.join(EMB,'co_cis_embeddings.npy'),allow_pickle=True)))
    CO=np.array([co_map[f"{e}__{p}"] for e,p in zip(df['enhancer_id'],df['promoter_id'])])

    reds={
      'ABC(基线)': None,
      'ABC+coembed(KPCA100 无监督)': mk_kpca(100),
    }
    for k in [5,10,20,50]: reds[f'ABC+coembed(PLS{k})']=mk_pls(k)
    reds['ABC+coembed(LDA shrink)']=mk_lda()
    print(f"{'配置':34s} {'AUC':>12s} {'PR-AUC':>12s}")
    res={}
    for nm,r in reds.items():
        m,s,mp=cv_gb(ABC,CO,y,groups,r); res[nm]={'auc_m':float(m),'auc_s':float(s),'ap_m':float(mp)}
        print(f"{nm:34s} {m:.4f}±{s:.4f}  {mp:.4f}",flush=True)
    with open(os.path.join(OUT,"cn_supervised_dr.json"),"w",encoding="utf-8") as f: json.dump(res,f,indent=2,ensure_ascii=False)
    print("\n→ cn_supervised_dr.json")
if __name__=='__main__': main()
