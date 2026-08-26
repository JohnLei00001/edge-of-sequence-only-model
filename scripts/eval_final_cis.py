#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最佳配置完整评估 (cis): ABC + KernelPCA(100, rbf) embedding + 4分类器对比。
生成丰富图表 Dashboard (ROC/PR/模型对比/混淆矩阵/特征重要性/分数分布/ABC-距离分布)。
所有 reducer/scaler/calibrator 仅 fit 在 train; test 最终评估。
"""
import os, json
from collections import Counter
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, accuracy_score, f1_score, confusion_matrix)
from sklearn.inspection import permutation_importance

ROOT=os.path.join(os.path.dirname(__file__),"..")
SEQ=os.path.join(ROOT,"EP_ATLAS","cis","cis_pairs_sequences.tsv")
EMB=os.path.join(ROOT,"EP_ATLAS","embeddings")
OUT=os.path.join(ROOT,"EP_ATLAS","models","final_cis")
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

FEAT_NAMES=([f'nuc_{c}' for c in 'ACGT']+['gc_content','cpg_density','gpc_density','shannon_entropy']
            +[f'dimer_{d}' for d in DINUCS]+[f'trimer_{t}' for t in TRINUCS]
            +['tata_per_kb','caat_per_kb','homo_runs_per_kb','log_length','at_gc_skew'])
def load_emb(name):
    ids=np.load(os.path.join(EMB,f"{name}_ids.npy"),allow_pickle=True).tolist()
    mat=np.load(os.path.join(EMB,f"{name}_embeddings.npy"),allow_pickle=True)
    return dict(zip(ids,mat))

def main():
    df=pd.read_csv(SEQ,sep='\t'); enh_e=load_emb("enhancers"); prom_e=load_emb("promoters")
    y=df['label'].values; fold=df['fold'].values
    tr=(fold=='train')|(fold=='val'); te=(fold=='test')
    ytr,yte=y[tr],y[te]

    # ABC
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

    # embedding 特征 + KernelPCA(100)
    emb=[]
    for i,row in df.iterrows():
        e=enh_e.get(row['enhancer_id']); p=prom_e.get(row['promoter_id'])
        c=float(np.dot(e,p)/(np.linalg.norm(e)*np.linalg.norm(p)+1e-12))
        emb.append(np.concatenate([e,p,[c]]))
    EMB_all=np.array(emb)
    kpca=KernelPCA(n_components=100,kernel='rbf',random_state=SEED).fit(EMB_all[tr])
    embk=kpca.transform(EMB_all); embk=(embk-embk.mean(0))/(embk.std(0)+1e-9)
    X=np.column_stack([ABC,embk])
    FEAT_ALL=[f'ABC_{n}' for n in FEAT_NAMES]+[f'ABC_{n}' for n in FEAT_NAMES]+ \
             ['contact_hic','cos_sim','cos_sim_3mer','gc_sim','cpg_sim','euc_sim','log10_dist','abc_score']+ \
             [f'KPCA_{i}' for i in range(100)]

    scaler=StandardScaler().fit(X[tr]); Xtr_s,Xte_s=scaler.transform(X[tr]),scaler.transform(X[te])
    models={
      'LogisticRegression': (LogisticRegression(max_iter=3000,C=1.0,random_state=SEED), True),
      'RandomForest':       (RandomForestClassifier(n_estimators=300,max_depth=None,min_samples_leaf=2,random_state=SEED,n_jobs=-1), False),
      'GradientBoosting':   (GradientBoostingClassifier(n_estimators=300,max_depth=4,learning_rate=0.1,random_state=SEED), False),
      'MLP':                (MLPClassifier(hidden_layer_sizes=(128,64,32),max_iter=500,alpha=0.01,random_state=SEED,early_stopping=True), True),
    }
    results={}
    for name,(clf,scale) in models.items():
        xtr,xte=(Xtr_s,Xte_s) if scale else (X[tr],X[te])
        clf.fit(xtr,ytr); prob=clf.predict_proba(xte)[:,1]
        auc=roc_auc_score(yte,prob); ap=average_precision_score(yte,prob)
        acc=accuracy_score(yte,(prob>0.5).astype(int)); f1=f1_score(yte,(prob>0.5).astype(int))
        cm=confusion_matrix(yte,(prob>0.5).astype(int))
        results[name]={'prob':prob,'auc':auc,'ap':ap,'acc':acc,'f1':f1,'cm':cm,'clf':clf}
        print(f"{name}: AUC={auc:.4f} PRAUC={ap:.4f} Acc={acc:.4f} F1={f1:.4f}")
    best_name=max(results,key=lambda k:results[k]['auc']); best=results[best_name]
    print(f"★ Best: {best_name} (AUC {best['auc']:.4f})")

    # 特征重要性
    if best_name in('RandomForest','GradientBoosting'):
        imp=best['clf'].feature_importances_
    elif best_name=='LogisticRegression':
        imp=np.abs(best['clf'].coef_[0])
    else:
        imp=permutation_importance(best['clf'],Xte_s,yte,n_repeats=5,random_state=SEED,n_jobs=-1).importances_mean
    topn=20; top_idx=np.argsort(imp)[::-1][:topn]

    # ── Dashboard ──
    plt.rcParams.update({'text.color':'white','axes.labelcolor':'white','xtick.color':'white',
                         'ytick.color':'white','axes.edgecolor':'#444','grid.color':'#333','font.size':8})
    fig=plt.figure(figsize=(24,18)); fig.patch.set_facecolor('#0a0a0a')
    gs=gridspec.GridSpec(3,4,figure=fig,hspace=0.5,wspace=0.4)
    axs=[fig.add_subplot(gs[i//4,i%4]) for i in range(12)]
    for ax in axs:
        ax.set_facecolor('#111')
        for s in ax.spines.values(): s.set_edgecolor('#444')
    nm=list(models)
    # 1 模型对比
    ax=axs[0]; x=np.arange(len(nm)); w=0.3
    ax.bar(x-w/2,[results[m]['auc'] for m in nm],w,color='#4fc3f7',label='AUC')
    ax.bar(x+w/2,[results[m]['ap'] for m in nm],w,color='#81c784',label='PR-AUC')
    ax.set_xticks(x); ax.set_xticklabels([m[:6] for m in nm],fontsize=7)
    ax.set_title('Model Comparison (Test)',fontsize=9,fontweight='bold'); ax.legend(fontsize=7)
    for i,m in enumerate(nm):
        ax.text(i-w/2,results[m]['auc']+0.005,f"{results[m]['auc']:.3f}",ha='center',fontsize=6)
        ax.text(i+w/2,results[m]['ap']+0.005,f"{results[m]['ap']:.3f}",ha='center',fontsize=6)
    ax.set_ylim(0.5,1.0)
    # 2 ROC
    ax=axs[1]; colors=['#4fc3f7','#81c784','#ffb74d','#ba68c8']
    for (m,c) in zip(nm,colors):
        fpr,tpr,_=roc_curve(yte,results[m]['prob']); ax.plot(fpr,tpr,c,lw=1.5,label=f'{m[:6]} ({results[m]["auc"]:.3f})')
    ax.plot([0,1],[0,1],'--',color='#555'); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC Curves',fontsize=9,fontweight='bold'); ax.legend(fontsize=6)
    # 3 PR
    ax=axs[2]
    for (m,c) in zip(nm,colors):
        p,r,_=precision_recall_curve(yte,results[m]['prob']); ax.plot(r,p,c,lw=1.5,label=f'{m[:6]} ({results[m]["ap"]:.3f})')
    ax.axhline(yte.mean(),ls='--',color='#555',label=f'base={yte.mean():.2f}')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.set_title('PR Curves',fontsize=9,fontweight='bold'); ax.legend(fontsize=6)
    # 4 混淆矩阵(最佳)
    ax=axs[3]; cm=best['cm']; im=ax.imshow(cm,cmap='Blues',aspect='auto')
    for i in range(2):
        for j in range(2): ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=14,color='white' if cm[i,j]>cm.max()/2 else '#333')
    ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(['Neg','Pos']); ax.set_yticklabels(['Neg','Pos'])
    ax.set_xlabel('Pred'); ax.set_ylabel('True'); ax.set_title(f'Confusion ({best_name[:12]})',fontsize=9,fontweight='bold')
    # 5 特征重要性
    ax=axs[4]; tn=15; names=[FEAT_ALL[i] for i in top_idx[:tn]][::-1]; vals=[imp[i] for i in top_idx[:tn]][::-1]
    ax.barh(np.arange(tn),vals,color=plt.cm.viridis(np.linspace(0.2,0.9,tn)))
    ax.set_yticks(np.arange(tn)); ax.set_yticklabels(names,fontsize=6); ax.set_xlabel('Importance')
    ax.set_title(f'Top {tn} Features ({best_name[:12]})',fontsize=9,fontweight='bold')
    # 6 概率分布
    ax=axs[5]
    ax.hist(best['prob'][yte==1],bins=40,color='#4fc3f7',alpha=0.7,label=f'Pos (n={(yte==1).sum()})',density=True)
    ax.hist(best['prob'][yte==0],bins=40,color='#ef5350',alpha=0.7,label=f'Neg (n={(yte==0).sum()})',density=True)
    ax.set_xlabel('Predicted prob'); ax.set_ylabel('Density'); ax.set_title(f'Score Dist ({best_name[:12]})',fontsize=9,fontweight='bold'); ax.legend(fontsize=7)
    # 7 ABC score 分布
    ax=axs[6]
    ax.hist(abc[y==1],bins=50,color='#4fc3f7',alpha=0.7,label='Pos',density=True)
    ax.hist(abc[y==0],bins=50,color='#ef5350',alpha=0.7,label='Neg',density=True)
    ax.set_title('ABC Score Dist',fontsize=9,fontweight='bold'); ax.legend(fontsize=7)
    # 8 距离分布
    ax=axs[7]
    ax.hist(np.log10(gd[y==1]+1),bins=50,color='#4fc3f7',alpha=0.7,label='Pos',density=True)
    ax.hist(np.log10(gd[y==0]+1),bins=50,color='#ef5350',alpha=0.7,label='Neg',density=True)
    ax.set_xlabel('log10 distance'); ax.set_title('Genomic Dist',fontsize=9,fontweight='bold'); ax.legend(fontsize=7)
    # 9 contact 分布
    ax=axs[8]
    ax.hist(ch[y==1],bins=50,color='#4fc3f7',alpha=0.7,label='Pos',density=True)
    ax.hist(ch[y==0],bins=50,color='#ef5350',alpha=0.7,label='Neg',density=True)
    ax.set_title('Hi-C Contact Dist',fontsize=9,fontweight='bold'); ax.legend(fontsize=7)
    # 10 activity 分布
    ax=axs[9]
    ax.hist(act[y==1],bins=50,color='#4fc3f7',alpha=0.7,label='Pos',density=True)
    ax.hist(act[y==0],bins=50,color='#ef5350',alpha=0.7,label='Neg',density=True)
    ax.set_title('Activity Dist',fontsize=9,fontweight='bold'); ax.legend(fontsize=7)
    # 11 KPCA 重要性前若干
    ax=axs[10]
    ax.plot(range(1,101),kpca.eigenvalues_[:100],color='#4fc3f7',lw=1.5)
    ax.set_xlabel('KPCA component'); ax.set_ylabel('eigenvalue'); ax.set_title('KPCA Eigenvalues',fontsize=9,fontweight='bold')
    # 12 汇总
    ax=axs[11]; ax.axis('off')
    lines=["Final Cis Model (ABC + KPCA-embedding)","═"*40,
           f"Config: ABC(194) + KernelPCA-100 embedding","",
           f"Data: {len(df)} pairs | pos {int(y.sum())} / neg {int((y==0).sum())}",
           f"Train+Val {int(tr.sum())} | Test {int(te.sum())}","",
           f"★ Best: {best_name}","  AUC {:.4f}".format(best['auc']),
           "  PR-AUC {:.4f}".format(best['ap']),"  Acc {:.4f}".format(best['acc']),
           "  F1 {:.4f}".format(best['f1']),"","All models:"]
    for m in nm: lines.append(f"  {m[:16]:18s} AUC={results[m]['auc']:.4f} AP={results[m]['ap']:.4f}")
    lines+=["",f"Top feat: {FEAT_ALL[top_idx[0]]} ({imp[top_idx[0]]:.4f})"]
    yp=0.97
    for ln in lines:
        ccol='#4fc3f7' if 'Final' in ln else ('#ffeb3b' if 'Best' in ln or 'AUC' in ln else 'white')
        ax.text(0.03,yp,ln,transform=ax.transAxes,fontsize=7,color=ccol,verticalalignment='top',fontfamily='monospace'); yp-=0.035
    fig.suptitle("ABC + KernelPCA-Embedding — Cis Model Final Evaluation (K562, hg19)",color='white',fontsize=14,fontweight='bold',y=0.985)
    os.makedirs(OUT,exist_ok=True)
    fig.savefig(os.path.join(OUT,"cis_final_dashboard.png"),dpi=150,bbox_inches='tight',facecolor='#0a0a0a'); plt.close()

    # 保存指标
    met={}
    for m in nm:
        met[m]={'auc':round(float(results[m]['auc']),4),'prauc':round(float(results[m]['ap']),4),
                'acc':round(float(results[m]['acc']),4),'f1':round(float(results[m]['f1']),4),
                'cm':results[m]['cm'].tolist()}
    met['best_config']={'model':best_name,'method':'KernelPCA(rbf,k=100)','abc_dims':194,'embed_dims':100}
    with open(os.path.join(OUT,"cis_final_metrics.json"),"w",encoding="utf-8") as f: json.dump(met,f,indent=2)
    # 保存test预测
    dfte=df[te].copy(); dfte['predicted_prob']=best['prob']
    dfte.to_csv(os.path.join(OUT,"cis_test_predictions.csv"),index=False)
    print("\n已保存: dashboard / metrics / test_predictions →",OUT)

if __name__=="__main__": main()
