#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强子-启动子互作二分类：cis 与 trans 分开训练。
模型：SVM(RBF) / RandomForest / XGBoost
严谨流程：
  - 按 meta.fold 划分 train/val/test
  - SVM 用 StandardScaler(仅 fit 在 train, 避免泄漏); RF/XGB 不缩放
  - 在 val 上网格搜索调超参(以 val-AUC 选优)
  - 最优模型在 train 上拟合, 在 test 上做最终评估
指标: AUC / PR-AUC / Accuracy / F1 / 混淆矩阵 + train-AUC(过拟合参考)
"""
import os, csv, json, time
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, f1_score, confusion_matrix)
from xgboost import XGBClassifier

ROOT = os.path.join(os.path.dirname(__file__), "..")
FEAT = os.path.join(ROOT, "EP_ATLAS", "features")
OUT  = os.path.join(ROOT, "EP_ATLAS", "models")
SEED = 42

def load_group(g):
    X = np.load(os.path.join(FEAT, f"{g}_X.npy"))
    y = np.load(os.path.join(FEAT, f"{g}_y.npy"))
    meta = list(csv.DictReader(open(os.path.join(FEAT, f"{g}_meta.tsv")), delimiter="\t"))
    folds = np.array([m["fold"] for m in meta])
    tr, va, te = (np.where(folds == f)[0] for f in ("train", "val", "test"))
    return X, y, tr, va, te

def eval_model(clf, Xtr, ytr, Xte, yte):
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = clf.predict(Xte)
    auc = roc_auc_score(yte, proba)
    prauc = average_precision_score(yte, proba)
    acc = accuracy_score(yte, pred)
    f1 = f1_score(yte, pred)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    # train AUC (过拟合参考)
    trauc = roc_auc_score(ytr, clf.predict_proba(Xtr)[:, 1])
    return dict(AUC=auc, PRAUC=prauc, Acc=acc, F1=f1,
                TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn), TrainAUC=trauc)

def grid_search(model_builder, params_grid, Xtr, ytr, Xva, yva, scale=False):
    scaler = StandardScaler().fit(Xtr) if scale else None
    def prep(A):
        return scaler.transform(A) if scale else A
    Xt, Xv = prep(Xtr), prep(Xva)
    best, best_auc = None, -1
    import itertools
    keys = list(params_grid)
    for combo in itertools.product(*params_grid.values()):
        hp = dict(zip(keys, combo))
        clf = model_builder(**hp)
        clf.fit(Xt, ytr)
        a = roc_auc_score(yva, clf.predict_proba(Xv)[:, 1])
        if a > best_auc:
            best_auc, best = a, hp
    return best, best_auc

def build_svm(C=10, gamma="scale"):
    return SVC(C=C, gamma=gamma, probability=True, random_state=SEED)

def build_rf(n_estimators=300, max_depth=None, min_samples_leaf=2):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf,
                                  n_jobs=-1, random_state=SEED)

def build_xgb(n_estimators=300, max_depth=4, learning_rate=0.1):
    return XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                         learning_rate=learning_rate, eval_metric="auc",
                         n_jobs=-1, random_state=SEED)

def run_group(g):
    X, y, tr, va, te = load_group(g)
    print(f"\n========== {g} ==========")
    print(f"样本: train={len(tr)} val={len(va)} test={len(te)} 特征={X.shape[1]}")
    report = {"group": g, "n_train": len(tr), "n_val": len(va), "n_test": len(te)}
    models = {
        "SVM":  (build_svm, dict(C=[1, 10, 100], gamma=["scale", 0.001]), True),
        "RF":   (build_rf, dict(n_estimators=[200, 400], max_depth=[None, 20], min_samples_leaf=[1, 4]), False),
        "XGB":  (build_xgb, dict(n_estimators=[200, 400], max_depth=[3, 6], learning_rate=[0.05, 0.1]), False),
    }
    for name, (builder, grid, scale) in models.items():
        t0 = time.time()
        best, vauc = grid_search(builder, grid, X[tr], y[tr], X[va], y[va], scale=scale)
        clf = builder(**best)
        m = eval_model(clf, X[tr], y[tr], X[te], y[te])
        print(f"  {name:4s}  best={best} valAUC={vauc:.4f} | "
              f"test: AUC={m['AUC']:.4f} PRAUC={m['PRAUC']:.4f} Acc={m['Acc']:.4f} "
              f"F1={m['F1']:.4f} (TP{int(m['TP'])} FP{int(m['FP'])} FN{int(m['FN'])} TN{int(m['TN'])}) "
              f"trainAUC={m['TrainAUC']:.4f} [{time.time()-t0:.0f}s]")
        report[f"{name}_best"] = best
        report[f"{name}_valAUC"] = round(float(vauc), 4)
        report[f"{name}_test"] = {k: round(float(v), 4) if isinstance(v, (int, float)) else v for k, v in m.items()}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{g}_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告 → {os.path.join(OUT, f'{g}_report.json')}")

if __name__ == "__main__":
    run_group("cis")
    run_group("trans")
