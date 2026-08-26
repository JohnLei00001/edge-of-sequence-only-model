#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最终评估: 显著性检验 + 可视化。收集关键配置的OOF预测, 配对bootstrap(AUC/Acc)+McNemar, 出图。"""
import os, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             roc_curve, precision_recall_curve, confusion_matrix)

ROOT=os.path.join(os.path.dirname(__file__),"..")
CACHE=os.path.join(ROOT,"EP_ATLAS","cache"); OUT=os.path.join(ROOT,"EP_ATLAS","results")
SEED=42; NBOOT=10000
C={'blue':'#0072B2','red':'#D55E00','green':'#009E73','purple':'#CC79A7','orange':'#E69F00','grey':'#999999'}
rcParams.update({'font.size':8,'axes.linewidth':0.8,'font.family':'sans-serif','figure.facecolor':'white',
 'axes.facecolor':'white','savefig.facecolor':'white'})

def load_cache():
    ABC=np.load(os.path.join(CACHE,'ABC.npy')); CO=np.load(os.path.join(CACHE,'CO.npy'))
    y=np.load(os.path.join(CACHE,'y.npy')); groups=np.load(os.path.join(CACHE,'groups.npy'),allow_pickle=True)
    return ABC,CO,y,groups

def oof_and_importances(cfg,ABC,CO,y,groups):
    """返回 (oof_prob, feature_importances或None, model)。cfg: 'ABC'|'ABC+co'|'ABC+coRED'"""
    oof=np.zeros(len(y))
    importances=[]
    for tr,va in GroupKFold(5).split(ABC,y,groups):
        if cfg=='ABC': X=ABC
        elif cfg=='ABC+co': X=np.column_stack([ABC,CO])
        else:
            kp=KernelPCA(n_components=100,kernel='rbf',random_state=SEED); kp.fit(CO[tr])
            red=kp.transform(CO); red=(red-red.mean(0))/(red.std(0)+1e-9); X=np.column_stack([ABC,red])
        sc=StandardScaler().fit(X[tr])
        clf=GradientBoostingClassifier(n_estimators=300,max_depth=4,learning_rate=0.1,random_state=SEED)
        clf.fit(sc.transform(X[tr]),y[tr]); oof[va]=clf.predict_proba(sc.transform(X[va]))[:,1]
        importances.append(clf.feature_importances_)
    return oof, np.mean(importances,axis=0)

def bootstrap_diff(y,pA,pB,NBOOT=10000,seed=0):
    """bootstrap AUC差(A-B)和Acc差(A-B) 的 CI 与单侧p"""
    rng=np.random.RandomState(seed); n=len(y); idx=np.arange(n)
    dAUC=np.zeros(NBOOT); dACC=np.zeros(NBOOT)
    for b in range(NBOOT):
        s=rng.choice(idx,size=n,replace=True)
        dAUC[b]=roc_auc_score(y[s],pA[s])-roc_auc_score(y[s],pB[s])
        dACC[b]=accuracy_score(y[s],(pA[s]>0.5).astype(int))-accuracy_score(y[s],(pB[s]>0.5).astype(int))
    return (np.percentile(dAUC,[2.5,97.5]), float(np.mean(dAUC<=0)),
            np.percentile(dACC,[2.5,97.5]), float(np.mean(dACC<=0)))

def main():
    ABC,CO,y,groups=load_cache()
    os.makedirs(OUT,exist_ok=True)
    print("收集 OOF 预测...",flush=True)
    preds={}
    for name in ['ABC','ABC+co','ABC+coRED']:
        t0=time.time(); p,_=oof_and_importances(name,ABC,CO,y,groups); preds[name]=p
        np.savez(os.path.join(OUT,'oof_predictions.npz'),**preds,y=y)  # 算完立即落盘,防丢失
        print(f"  {name}: OOF AUC={roc_auc_score(y,p):.4f} [{time.time()-t0:.0f}s]",flush=True)
    np.savez(os.path.join(OUT,'oof_predictions.npz'),**preds,y=y)

    # 显著性检验
    print("\n=== 配对显著性 (bootstrap 95%CI + 单侧p) ===")
    pairs=[('ABC+co','ABC'),('ABC+coRED','ABC'),('ABC+co','ABC+coRED')]
    sig={}
    for a,b in pairs:
        cA,cB=preds[a],preds[b]
        aucA,aucB=roc_auc_score(y,cA),roc_auc_score(y,cB)
        accA=accuracy_score(y,(cA>0.5).astype(int)); accB=accuracy_score(y,(cB>0.5).astype(int))
        ciA,pA,ciAcc,pAcc=bootstrap_diff(y,cA,cB)
        print(f"  {a} vs {b}: AUC {aucA:.4f} vs {aucB:.4f} (Δ={aucA-aucB:+.4f} [{ciA[0]:+.4f},{ciA[1]:+.4f}], p={pA:.4f})  Acc {accA:.4f} vs {accB:.4f} (p={pAcc:.4f})",flush=True)
        sig[f'{a}__vs__{b}']={'aucA':float(aucA),'aucB':float(aucB),'auc_ci':[float(x) for x in ciA],'auc_p':float(pA),
                              'accA':float(accA),'accB':float(accB),'acc_ci':[float(x) for x in ciAcc],'acc_p':float(pAcc)}
    json.dump(sig,open(os.path.join(OUT,'significance.json'),'w'),ensure_ascii=False,indent=2)

    # === 可视化 ===
    print("\n生成图表...",flush=True)
    CNAME={'ABC':'ABC (pure)','ABC+co':'ABC + co-embedding (raw)','ABC+coRED':'ABC + co-embedding (RED)'}
    COL={'ABC':C['grey'],'ABC+co':C['blue'],'ABC+coRED':C['red']}
    # ROC
    fig,ax=plt.subplots(figsize=(4,3.5))
    for nm in ['ABC','ABC+co','ABC+coRED']:
        fpr,tpr,_=roc_curve(y,preds[nm]); ax.plot(fpr,tpr,color=COL[nm],lw=1.5,label=f"{CNAME[nm]} ({roc_auc_score(y,preds[nm]):.3f})")
    ax.plot([0,1],[0,1],'--',color='#999'); ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate'); ax.legend(frameon=False,fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig_roc.png'),dpi=300); plt.close(fig)
    # PR
    fig,ax=plt.subplots(figsize=(4,3.5))
    for nm in ['ABC','ABC+co','ABC+coRED']:
        pr,rc,_=precision_recall_curve(y,preds[nm]); ax.plot(rc,pr,color=COL[nm],lw=1.5,label=f"{CNAME[nm]} ({average_precision_score(y,preds[nm]):.3f})")
    ax.axhline(y.mean(),ls='--',color='#999'); ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.legend(frameon=False,fontsize=6.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig_pr.png'),dpi=300); plt.close(fig)
    # 混淆矩阵 (ABC+co)
    fig,ax=plt.subplots(figsize=(3,2.8)); cm=confusion_matrix(y,(preds['ABC+co']>0.5).astype(int))
    ax.imshow(cm,cmap='Blues',aspect='auto')
    for i in range(2):
        for j in range(2): ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=11)
    ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(['Neg','Pos']); ax.set_yticklabels(['Neg','Pos'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('ABC+coembed (raw) GB',fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig_confusion.png'),dpi=300); plt.close(fig)
    # 特征重要性 (ABC+co, 训练全量)
    t0=time.time(); print("训练特征重要性模型...",flush=True)
    X=np.column_stack([ABC,CO]); sc=StandardScaler().fit(X)
    clf=GradientBoostingClassifier(n_estimators=300,max_depth=4,learning_rate=0.1,random_state=SEED)
    clf.fit(sc.transform(X),y); imp=clf.feature_importances_
    nABC=ABC.shape[1]
    abc_share, co_share = imp[:nABC].sum(), imp[nABC:].sum()
    print(f"  特征重要性: ABC部分占比={abc_share:.3f}, coembed部分占比={co_share:.3f}",flush=True)
    topn=20; ti=np.argsort(imp)[::-1][:topn]
    names=[f'ABC#{i}' if i<nABC else f'co#{i-nABC}' for i in range(len(imp))]
    fig,ax=plt.subplots(figsize=(4,4))
    idx=ti[::-1]; ax.barh(np.arange(topn),imp[idx],color=[C['blue'] if i<nABC else C['red'] for i in idx])
    ax.set_yticks(np.arange(topn)); ax.set_yticklabels([names[i] for i in idx],fontsize=6); ax.set_xlabel('Feature importance')
    ax.set_title(f'Top {topn} features (blue=ABC, red=coembed)',fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig_feature_importance.png'),dpi=300); plt.close(fig)
    # ABC vs coembed 重要性占比饼图
    fig,ax=plt.subplots(figsize=(3,3)); ax.pie([abc_share,co_share],labels=['ABC features','co-embedding'],autopct='%1.1f%%',colors=[C['blue'],C['red']],startangle=90)
    ax.set_title('Importance share',fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig_importance_share.png'),dpi=300); plt.close(fig)
    # 配置对比柱状图(读compare_8configs)
    import json as J
    d8=J.load(open(os.path.join(OUT,'compare_8configs.json'),encoding='utf-8'))
    cfgs=[v for v in d8.values() if v['model']=='GB']
    cfgs.sort(key=lambda v:-v['auc_m'])
    fig,ax=plt.subplots(figsize=(6,3.5)); x=np.arange(len(cfgs))
    ax.bar(x,[v['auc_m'] for v in cfgs],color=C['blue'],alpha=0.85)
    ax.axhline(0.5,ls='--',color='#999',lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([v['config'] for v in cfgs],rotation=40,ha='right',fontsize=6)
    ax.set_ylabel('AUC (grouped CV)'); ax.set_title('GB configs comparison',fontsize=8); ax.set_ylim(0.5,0.75)
    for i,v in enumerate(cfgs): ax.text(i,v['auc_m']+0.005,f"{v['auc_m']:.3f}",ha='center',fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(OUT,'fig_config_compare.png'),dpi=300); plt.close(fig)
    print("\n图表已生成 →",OUT)
    print("留痕: oof_predictions.npz / significance.json / fig_*.png")

if __name__=='__main__': main()
