#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABC vs ABC+coembed 配对显著性检验。
收集分组5折CV的 out-of-fold 预测 → bootstrap AUC差置信区间 + 单侧p值 + 逐折配对Wilcoxon。
"""
import os, json
from collections import Counter
import numpy as np, pandas as pd
from scipy import stats
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

ROOT=os.path.join(os.path.dirname(__file__),"..")
SEQ=os.path.join(ROOT,"EP_ATLAS","cis","cis_pairs_sequences.tsv")
EMB=os.path.join(ROOT,"EP_ATLAS","embeddings"); OUT=os.path.join(ROOT,"EP_ATLAS","report")
SEED=42; np.random.seed(SEED); NBOOT=10000
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
    mat=np.load(os.path.join(EMB,f"{name}_embeddings.npy"),allow_pickle=True); return dict(zip(ids,mat))
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

def oof_predictions(X,y,groups):
    """返回每个样本的 out-of-fold 概率。"""
    oof=np.zeros(len(y))
    for tr,va in GroupKFold(5).split(X,y,groups):
        sc=StandardScaler().fit(X[tr]); Xtr,Xva=sc.transform(X[tr]),sc.transform(X[va])
        clf=GradientBoostingClassifier(n_estimators=300,max_depth=4,learning_rate=0.1,random_state=SEED)
        clf.fit(Xtr,y[tr]); oof[va]=clf.predict_proba(Xva)[:,1]
    return oof

def main():
    df=pd.read_csv(SEQ,sep='\t'); y=df['label'].values; groups=df['enhancer_id'].values
    tr_all=np.arange(len(df)); ABC=build_abc(df,tr_all,y)
    co_map=dict(zip(np.load(os.path.join(EMB,'co_cis_ids.npy'),allow_pickle=True).tolist(),np.load(os.path.join(EMB,'co_cis_embeddings.npy'),allow_pickle=True)))
    CO=np.array([co_map[f"{e}__{p}"] for e,p in zip(df['enhancer_id'],df['promoter_id'])])

    print("收集 out-of-fold 预测...")
    p_abc=oof_predictions(ABC,y,groups)
    # ABC+coembed: 每折内KPCA-100
    p_co=np.zeros(len(y))
    for tr,va in GroupKFold(5).split(ABC,y,groups):
        kp=KernelPCA(n_components=100,kernel='rbf',random_state=SEED).fit(CO[tr]); red=kp.transform(CO)
        red=(red-red.mean(0))/(red.std(0)+1e-9); X=np.column_stack([ABC,red])
        sc=StandardScaler().fit(X[tr]); clf=GradientBoostingClassifier(n_estimators=300,max_depth=4,learning_rate=0.1,random_state=SEED)
        clf.fit(sc.transform(X[tr]),y[tr]); p_co[va]=clf.predict_proba(sc.transform(X[va]))[:,1]
    auc_abc,auc_co=roc_auc_score(y,p_abc),roc_auc_score(y,p_co)
    diff=auc_co-auc_abc
    print(f"ABC AUC={auc_abc:.4f}   ABC+coembed AUC={auc_co:.4f}   Δ={diff:+.4f}")

    # bootstrap 配对 AUC差
    rng=np.random.RandomState(0); n=len(y); idx=np.arange(n)
    diffs=np.zeros(NBOOT)
    for b in range(NBOOT):
        s=rng.choice(idx,size=n,replace=True)
        diffs[b]=roc_auc_score(y[s],p_co[s])-roc_auc_score(y[s],p_abc[s])
    lo,hi=np.percentile(diffs,[2.5,97.5])
    pval_one=np.mean(diffs<=0)   # 单侧: P(Δ≤0)
    print(f"Bootstrap Δ 95%CI=[{lo:+.4f},{hi:+.4f}]  单侧p(Δ≤0)={pval_one:.4f}")

    # 逐折配对 Wilcoxon
    fold_auc={'ABC':[],'co':[]}
    for tr,va in GroupKFold(5).split(ABC,y,groups):
        fold_auc['ABC'].append(roc_auc_score(y[va],p_abc[va]))
        fold_auc['co'].append(roc_auc_score(y[va],p_co[va]))
    w,pw=stats.wilcoxon(fold_auc['ABC'],fold_auc['co'])
    print(f"逐折 AUC: ABC={[f'{x:.4f}' for x in fold_auc['ABC']]}  co={[f'{x:.4f}' for x in fold_auc['co']]}")
    print(f"配对Wilcoxon: p={pw:.4f}")
    json.dump({'auc_abc':float(auc_abc),'auc_co':float(auc_co),'diff':float(diff),
               'ci95':[float(lo),float(hi)],'p_one_sided':float(pval_one),'wilcoxon_p':float(pw)},
              open(os.path.join(OUT,'cn_significance_test.json'),'w'),indent=2)
    print("→ cn_significance_test.json")

if __name__=='__main__': main()
