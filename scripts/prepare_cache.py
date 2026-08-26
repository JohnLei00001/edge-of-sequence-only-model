#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""一次性缓存对齐的特征, 供后续实验复用并留痕:
  ABC(2526x194), CO(coembed 2526x1024), SEP(独立拼接 2526x2049),
  y, groups(enhancer_id), fold, coords. 存到 EP_ATLAS/cache/.
"""
import os, json
from collections import Counter
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

ROOT=os.path.join(os.path.dirname(__file__),"..")
SEQ=os.path.join(ROOT,"EP_ATLAS","cis","cis_pairs_sequences.tsv")
EMB=os.path.join(ROOT,"EP_ATLAS","embeddings")
CACHE=os.path.join(ROOT,"EP_ATLAS","cache")
SEED=42
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

def main():
    df=pd.read_csv(SEQ,sep='\t'); y=df['label'].values; groups=df['enhancer_id'].values; fold=df['fold'].values
    enh_mid=(df['enhancer_start']+df['enhancer_end']).values/2.0; prom_mid=(df['promoter_start']+df['promoter_end']).values/2.0
    gd=np.maximum(np.abs(enh_mid-prom_mid),1.0); ch=(gd**(-0.87))*np.exp(-gd/3e6); ch=(ch-ch.min())/(ch.max()-ch.min()+1e-10)
    log10d=np.log10(gd+1)
    enh_f=np.array([extract_features(s) for s in df['enhancer_sequence']]); prom_f=np.array([extract_features(s) for s in df['promoter_sequence']])
    def cos(a,b): return np.array([np.dot(a[i],b[i])/(np.linalg.norm(a[i])*np.linalg.norm(b[i])+1e-10) for i in range(len(a))])
    cos_sim=cos(enh_f,prom_f); cos_tri=cos(enh_f[:,21:85],prom_f[:,21:85])
    gc_sim=1-np.abs(enh_f[:,4]-prom_f[:,4]); cpg_sim=1/(1+np.abs(enh_f[:,5]-prom_f[:,5]))
    euc=np.linalg.norm(enh_f[:,:20]-prom_f[:,:20],axis=1); euc_sim=1/(1+euc)
    cf=np.column_stack([ch,cos_sim,cos_tri,gc_sim,cpg_sim,euc_sim])
    tr_all=np.arange(len(df))
    def calib(f):
        c=LogisticRegression(max_iter=1000,C=0.1,random_state=SEED); c.fit(f[tr_all],y[tr_all]); s=c.decision_function(f)
        return (s-s.min())/(s.max()-s.min()+1e-10)
    act=calib(PCA(10,random_state=SEED).fit_transform(enh_f)); cont=calib(cf)
    abc=np.zeros(len(df))
    for g,idx in df.groupby('gene_id').indices.items():
        num=act[idx]*cont[idx]; denom=num.sum()+1e-10; abc[idx]=num/denom
    ABC=np.column_stack([enh_f,prom_f,cf,log10d[:,None],abc[:,None]])
    # co-embedding
    co_map=dict(zip(np.load(os.path.join(EMB,'co_cis_ids.npy'),allow_pickle=True).tolist(),np.load(os.path.join(EMB,'co_cis_embeddings.npy'),allow_pickle=True)))
    CO=np.array([co_map[f"{e}__{p}"] for e,p in zip(df['enhancer_id'],df['promoter_id'])])
    # 独立拼接 embedding (enh+prom+cos)
    enh_e=load_emb("enhancers"); prom_e=load_emb("promoters")
    E=np.array([enh_e[e] for e in df['enhancer_id']]); P=np.array([prom_e[p] for p in df['promoter_id']])
    cos_sep=np.array([np.dot(E[i],P[i])/(np.linalg.norm(E[i])*np.linalg.norm(P[i])+1e-12) for i in range(len(E))])
    SEP=np.column_stack([E,P,cos_sep])
    os.makedirs(CACHE,exist_ok=True)
    for nm,arr in [('ABC',ABC),('CO',CO),('SEP',SEP)]: np.save(os.path.join(CACHE,f"{nm}.npy"),arr)
    np.save(os.path.join(CACHE,'y.npy'),y); np.save(os.path.join(CACHE,'groups.npy'),np.array(groups,dtype=object))
    np.save(os.path.join(CACHE,'fold.npy'),np.array(fold,dtype=object))
    json.dump({'ABC':ABC.shape,'CO':CO.shape,'SEP':SEP.shape,'n':len(y),'pos':int(y.sum()),'neg':int((y==0).sum()),
               'ABC构建参数':{'gamma':0.87,'D':3e6,'activity_PCA':10}},open(os.path.join(CACHE,'manifest.json'),'w'),ensure_ascii=False,indent=2)
    print(f"缓存完成: ABC{ABC.shape} CO{CO.shape} SEP{SEP.shape} n={len(y)} pos={int(y.sum())} neg={int((y==0).sum())}")
    print("→",CACHE)
if __name__=='__main__': main()
