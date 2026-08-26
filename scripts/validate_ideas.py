#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证 idea (轻量, 不深究分类器): GB 默认 + 分组5折CV。
① 探针诊断: 单独 embedding(co/sep) 在 train vs test 上的 AUC (信号是否存在)
② 残差化: 对 embedding 去中心化+去顶k主成分后, 与 ABC 融合
③ 交互特征: 独立 enh/prom embedding 的元素级 diff/prod 等交互项, 与 ABC 融合
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

def cv_gb(X,y,groups):
    aucs,aps=[],[]
    for tr,va in GroupKFold(5).split(X,y,groups):
        sc=StandardScaler().fit(X[tr]); Xtr,Xva=sc.transform(X[tr]),sc.transform(X[va])
        clf=GradientBoostingClassifier(n_estimators=300,max_depth=4,learning_rate=0.1,random_state=SEED)
        clf.fit(Xtr,y[tr]); p=clf.predict_proba(Xva)[:,1]
        aucs.append(roc_auc_score(y[va],p)); aps.append(average_precision_score(y[va],p))
    return np.mean(aucs),np.std(aucs),np.mean(aps)

def main():
    df=pd.read_csv(SEQ,sep='\t'); y=df['label'].values; groups=df['enhancer_id'].values
    tr_all=np.arange(len(df))
    ABC=build_abc(df,tr_all,y)
    enh_e=load_emb("enhancers"); prom_e=load_emb("promoters")
    # 独立 enh/prom embedding 按对
    E=np.array([enh_e[e] for e in df['enhancer_id']]); P=np.array([prom_e[p] for p in df['promoter_id']])
    # co-embedding
    coids=np.load(os.path.join(EMB,'co_cis_ids.npy'),allow_pickle=True).tolist()
    coemb=np.load(os.path.join(EMB,'co_cis_embeddings.npy'),allow_pickle=True)
    co_map=dict(zip(coids,coemb)); CO=np.array([co_map[f"{e}__{p}"] for e,p in zip(df['enhancer_id'],df['promoter_id'])])

    print("=== ① 探针诊断: 单独embedding train vs test AUC (单次切分) ===")
    from sklearn.model_selection import train_test_split
    for nm,M in [('coembed',CO),('sep-concat',np.concatenate([E,P],1))]:
        Xtr,Xte,ytr,yte=train_test_split(M,y,test_size=0.3,random_state=0,stratify=y)
        clf=GradientBoostingClassifier(n_estimators=200,max_depth=3,random_state=SEED)
        clf.fit(Xtr,ytr); tr_auc=roc_auc_score(ytr,clf.predict_proba(Xtr)[:,1]); te_auc=roc_auc_score(yte,clf.predict_proba(Xte)[:,1])
        print(f"  {nm:12s} trainAUC={tr_auc:.4f} testAUC={te_auc:.4f}  (train高test低=过拟合; 都低=无信号)")

    print("\n=== ②③ 残差化 + 交互特征 (分组CV, GB) ===")
    res={}
    m,s,mp=cv_gb(ABC,y,groups); res['ABC']={'auc_m':m,'auc_s':s,'ap_m':mp}; print(f"{'ABC(基线)':28s} {m:.4f}±{s:.4f} PR={mp:.4f}")
    for k_remove in [5,10,20]:
        # 残差化: 对E/P各自去中心化+去顶k主成分
        def resid(M,k):
            c=M-M.mean(0)
            pca=PCA(n_components=k,random_state=SEED).fit(c)
            return c-c@pca.components_.T@pca.components_  # 投影掉顶k成分
        Er=resid(E,k_remove); Pr=resid(P,k_remove)
        diff=Er-Pr; prod=Er*Pr; absd=np.abs(diff)
        # 降维交互特征(去噪)
        IF=np.column_stack([diff,prod,absd])
        kp=KernelPCA(n_components=min(30,IF.shape[1]),kernel='rbf',random_state=SEED).fit(IF)
        IFr=kp.transform(IF); IFr=(IFr-IFr.mean(0))/(IFr.std(0)+1e-9)
        X=np.column_stack([ABC,IFr])
        m,s,mp=cv_gb(X,y,groups); res[f'ABC+resid-interact(k={k_remove})']={'auc_m':m,'auc_s':s,'ap_m':mp}
        print(f"{'ABC+残差交互(k'+str(k_remove)+')':28s} {m:.4f}±{s:.4f} PR={mp:.4f}")
        # coembed 残差
        Cor=resid(CO,k_remove)
        kp=KernelPCA(n_components=min(30,Cor.shape[1]),kernel='rbf',random_state=SEED).fit(Cor)
        Corr=kp.transform(Cor); Corr=(Corr-Corr.mean(0))/(Corr.std(0)+1e-9)
        Xc=np.column_stack([ABC,Corr])
        m,s,mp=cv_gb(Xc,y,groups); res[f'ABC+coembed-resid(k={k_remove})']={'auc_m':m,'auc_s':s,'ap_m':mp}
        print(f"{'ABC+coembed残差(k'+str(k_remove)+')':28s} {m:.4f}±{s:.4f} PR={mp:.4f}")
    with open(os.path.join(OUT,"cn_idea_validation.json"),"w",encoding="utf-8") as f:
        json.dump(res,f,indent=2,ensure_ascii=False)
    print("\n→ cn_idea_validation.json")

if __name__=='__main__': main()
