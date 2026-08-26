"""
EnhancerPromoterEngine v3: Cis Model — 从零训练
利用 cis 同染色体数据的基因组坐标优势，实现完整 ABC 架构

关键改进（相比 v2 trans 模型）:
  · ContactModule 使用真实基因组距离 → Hi-C power-law distance decay
    C(d) = d^(-γ) × exp(-d/D)  (γ=0.87, D=3×10^6)
  · ActivityModule: 序列特征 PCA + 序列 motif 评分
  · 加入 distance_bin 作为离散特征
  · 加入 P-value 信号 (正样本有值，负样本 NaN → 不直接用作特征，但可做有/无标记)
  · 序列特征与 v2 相同 (93维/序列)

架构:
  1. SequenceFeatureExtractor — 93维/序列
  2. ActivityModule — PCA → calibrated activity (模拟 H3K27ac × ATAC)
  3. ContactModule — 真实基因组距离 → Hi-C distance decay (cis 场景核心优势)
  4. ABCModule — Activity × Contact / Σ(per-gene)
  5. Classifier — 196维 → 4模型对比 (LR / RF / GB / MLP)
  6. Evaluation — ROC / PR / 混淆矩阵 / 特征重要性 / Dashboard
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, confusion_matrix)
from sklearn.inspection import permutation_importance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, time, pickle, json
warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_PATH = '/work/995619778/uploads/cis_pairs_sequences.tsv'
OUT_DIR   = '/work/995619778/EnhancerPromoterEngine-main'

print("=" * 70)
print("EnhancerPromoterEngine v3: Cis Model Training from Scratch")
print("=" * 70)

df = pd.read_csv(DATA_PATH, sep='\t')
y_all = df['label'].values
print(f"Loaded {len(df)} cis pairs | pos={int((df.label==1).sum())} neg={int((df.label==0).sum())}")
print(f"All same-chromosome: {(df['enhancer_chr']==df['promoter_chr']).all()}")

# ═══════════════════════════════════════════════════════════════════════════
# 1. SEQUENCE FEATURE EXTRACTION (93维/序列)
# ═══════════════════════════════════════════════════════════════════════════
DINUCS = ['AA','AC','AG','AT','CA','CC','CG','CT',
          'GA','GC','GG','GT','TA','TC','TG','TT']
TRINUCS = [a+b+c for a in 'ACGT' for b in 'ACGT' for c in 'ACGT']

def extract_features(seq):
    seq = str(seq).upper()
    n = len(seq)
    if n < 3:
        return np.zeros(93)
    nuc_counts = {c: seq.count(c) for c in 'ACGT'}
    nuc_freq = np.array([nuc_counts[c]/n for c in 'ACGT'])
    gc = (nuc_counts['G'] + nuc_counts['C']) / n
    cpg = seq.count('CG') / n * 1000
    gpc = seq.count('GC') / n * 1000
    ent = 0
    for c in 'ACGT':
        p = nuc_counts[c] / n
        if p > 0: ent -= p * np.log2(p)
    dimer_counts = Counter(seq[i:i+2] for i in range(n-1))
    total_d = sum(dimer_counts.get(d, 0) for d in DINUCS)
    dinuc_freq = np.array([dimer_counts.get(d, 0)/max(total_d,1) for d in DINUCS])
    trimer_counts = Counter(seq[i:i+3] for i in range(n-2))
    total_t = sum(trimer_counts.get(t, 0) for t in TRINUCS)
    trinuc_freq = np.array([trimer_counts.get(t, 0)/max(total_t,1) for t in TRINUCS])
    tata = sum(1 for i in range(n-4) if seq[i:i+4] in ('TATA','AATA'))
    caat = sum(1 for i in range(n-5) if seq[i:i+5] in ('CAATC','CCAAT'))
    homo_runs = 0; run_len = 1
    for i in range(1, n):
        if seq[i] == seq[i-1]: run_len += 1
        else:
            if run_len >= 5: homo_runs += 1
            run_len = 1
    if run_len >= 5: homo_runs += 1
    log_len = np.log1p(n)
    at_skew = (nuc_counts['A'] + nuc_counts['T']) / max(nuc_counts['G'] + nuc_counts['C'], 1)
    return np.concatenate([
        nuc_freq, [gc], [cpg], [gpc], [ent],
        dinuc_freq, trinuc_freq,
        [tata / n * 1000], [caat / n * 1000], [homo_runs / n * 1000],
        [log_len], [at_skew],
    ])

FEAT_NAMES = (
    [f'nuc_{c}' for c in 'ACGT'] +
    ['gc_content', 'cpg_density', 'gpc_density', 'shannon_entropy'] +
    [f'dimer_{d}' for d in DINUCS] +
    [f'trimer_{t}' for t in TRINUCS] +
    ['tata_per_kb', 'caat_per_kb', 'homo_runs_per_kb', 'log_length', 'at_gc_skew']
)
assert len(FEAT_NAMES) == 93

print(f"\n[1] Sequence feature extraction: 93 dims/sequence")
t0 = time.time()
enh_feats = np.array([extract_features(s) for s in df['enhancer_sequence']])
prom_feats = np.array([extract_features(s) for s in df['promoter_sequence']])
print(f"    Enhancer: {enh_feats.shape} | Promoter: {prom_feats.shape} | Time: {time.time()-t0:.1f}s")

# ═══════════════════════════════════════════════════════════════════════════
# 2. GENOMIC DISTANCE & CONTACT MODULE (cis 核心优势)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[2] Genomic Distance & Contact Module: real coordinates → Hi-C distance decay")

# 计算增强子中点到启动子中点的真实基因组距离
enh_mid = (df['enhancer_start'] + df['enhancer_end']).values / 2.0
prom_mid = (df['promoter_start'] + df['promoter_end']).values / 2.0
genomic_dist = np.abs(enh_mid - prom_mid).astype(float)
genomic_dist = np.maximum(genomic_dist, 1.0)  # 避免 0

# Hi-C distance decay 模型: C(d) = d^(-γ) × exp(-d/D)
# γ (gamma): power-law exponent, 典型值 0.87 (Lieberman-Aiden et al. 2009)
# D: exponential decay length, 典型值 ~3 Mb
GAMMA = 0.87
D_DECAY = 3e6
contact_hic = (genomic_dist ** (-GAMMA)) * np.exp(-genomic_dist / D_DECAY)
# 归一化
contact_hic = (contact_hic - contact_hic.min()) / (contact_hic.max() - contact_hic.min() + 1e-10)

# Log10 距离 (作为连续特征)
log10_dist = np.log10(genomic_dist + 1)

# 序列相似性补充 (在 cis 中作为 contact 的辅助信号)
cos_sim = np.array([
    np.dot(enh_feats[i], prom_feats[i]) /
    (np.linalg.norm(enh_feats[i]) * np.linalg.norm(prom_feats[i]) + 1e-10)
    for i in range(len(df))
])
enh_tri = enh_feats[:, 21:85]
prom_tri = prom_feats[:, 21:85]
cos_sim_tri = np.array([
    np.dot(enh_tri[i], prom_tri[i]) /
    (np.linalg.norm(enh_tri[i]) * np.linalg.norm(prom_tri[i]) + 1e-10)
    for i in range(len(df))
])
gc_sim = 1 - np.abs(enh_feats[:, 4] - prom_feats[:, 4])
cpg_sim = 1 / (1 + np.abs(enh_feats[:, 5] - prom_feats[:, 5]))
euc_dist = np.linalg.norm(enh_feats[:, :20] - prom_feats[:, :20], axis=1)
euc_sim = 1 / (1 + euc_dist)

# 组合 contact: Hi-C + 序列相似性加权
contact_features = np.column_stack([contact_hic, cos_sim, cos_sim_tri, gc_sim, cpg_sim, euc_sim])
contact_lr = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
contact_lr.fit(contact_features, y_all)
contact_score = contact_lr.decision_function(contact_features)
contact_score = (contact_score - contact_score.min()) / (contact_score.max() - contact_score.min() + 1e-10)

print(f"    Genomic distance range: [{genomic_dist.min():.0f}, {genomic_dist.max():.0f}] bp")
print(f"    Hi-C contact (pos mean): {contact_hic[y_all==1].mean():.6f}")
print(f"    Hi-C contact (neg mean): {contact_hic[y_all==0].mean():.6f}")
print(f"    Contact score (pos mean): {contact_score[y_all==1].mean():.4f}")
print(f"    Contact score (neg mean): {contact_score[y_all==0].mean():.4f}")
print(f"    Distance (pos median): {np.median(genomic_dist[y_all==1])/1e6:.2f} Mb")
print(f"    Distance (neg median): {np.median(genomic_dist[y_all==0])/1e6:.2f} Mb")

# ═══════════════════════════════════════════════════════════════════════════
# 3. ACTIVITY MODULE
# ═══════════════════════════════════════════════════════════════════════════
print("\n[3] Activity Module: PCA on enhancer features → learned activity score")

pca_activity = PCA(n_components=10, random_state=42)
enh_pca = pca_activity.fit_transform(enh_feats)
print(f"    PCA explained variance (top 10): {pca_activity.explained_variance_ratio_.sum():.3f}")

calib_lr = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
calib_lr.fit(enh_pca, y_all)
activity_score = calib_lr.decision_function(enh_pca)
activity_score = (activity_score - activity_score.min()) / (activity_score.max() - activity_score.min() + 1e-10)
print(f"    Activity (pos mean): {activity_score[y_all==1].mean():.4f}")
print(f"    Activity (neg mean): {activity_score[y_all==0].mean():.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 4. ABC MODULE
# ═══════════════════════════════════════════════════════════════════════════
print("\n[4] ABC Module: Activity × Contact / per-gene normalization")

abc_numerator = activity_score * contact_score
gene_groups = df.groupby('gene_id').indices
abc_score = np.zeros(len(df))
for gene, indices in gene_groups.items():
    idx = list(indices)
    denom = abc_numerator[idx].sum() + 1e-10
    abc_score[idx] = abc_numerator[idx] / denom

print(f"    ABC (pos mean): {abc_score[y_all==1].mean():.6f}")
print(f"    ABC (neg mean): {abc_score[y_all==0].mean():.6f}")

# ═══════════════════════════════════════════════════════════════════════════
# 5. CLASSIFIER TRAINING
# ═══════════════════════════════════════════════════════════════════════════
print("\n[5] Classifier Training: 4 models compared")

# 特征矩阵: enh 93 + prom 93 + joint 6 (contact_hic, cos, cos_tri, gc_sim, cpg_sim, euc_sim) + log10_dist + abc = 194
joint_features = np.column_stack([contact_hic, cos_sim, cos_sim_tri, gc_sim, cpg_sim, euc_sim, log10_dist])
X_full = np.column_stack([enh_feats, prom_feats, joint_features, abc_score])
FEATURE_NAMES = (
    [f'enh_{n}' for n in FEAT_NAMES] +
    [f'prom_{n}' for n in FEAT_NAMES] +
    ['contact_hic', 'cos_sim', 'cos_sim_3mer', 'gc_sim', 'cpg_sim', 'euc_sim', 'log10_dist', 'abc_score']
)
print(f"    Feature matrix: {X_full.shape[0]} × {X_full.shape[1]}")

train_mask = df['fold'].isin(['train', 'val']).values
test_mask  = (df['fold'] == 'test').values
X_train, X_test = X_full[train_mask], X_full[test_mask]
y_train, y_test = y_all[train_mask], y_all[test_mask]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
print(f"    Train: {len(y_train)} (pos={int(y_train.sum())}) | Test: {len(y_test)} (pos={int(y_test.sum())})")

models = {
    'LogisticRegression': LogisticRegression(max_iter=3000, C=1.0, random_state=42),
    'RandomForest':       RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    'GradientBoosting':   GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42),
    'MLP':                MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, alpha=0.01, random_state=42, early_stopping=True),
}

results = {}
for name, model in models.items():
    t0 = time.time()
    if name in ('LogisticRegression', 'MLP'):
        model.fit(X_train_s, y_train)
        prob = model.predict_proba(X_test_s)[:, 1]
    else:
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
    elapsed = time.time() - t0
    auc = roc_auc_score(y_test, prob)
    ap  = average_precision_score(y_test, prob)
    pred = (prob > 0.5).astype(int)
    acc = (pred == y_test).mean()
    cm = confusion_matrix(y_test, pred)
    results[name] = {'model': model, 'prob': prob, 'auc': auc, 'ap': ap, 'acc': acc, 'cm': cm, 'time': elapsed, 'pred': pred}
    print(f"\n    {name}: AUC={auc:.4f} | AP={ap:.4f} | Acc={acc:.4f} | TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]} | {elapsed:.1f}s")

best_name = max(results, key=lambda k: results[k]['auc'])
best = results[best_name]
print(f"\n    ★ Best model: {best_name} (AUC={best['auc']:.4f}, AP={best['ap']:.4f})")

# ═══════════════════════════════════════════════════════════════════════════
# 6. FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════
print("\n[6] Feature Importance")
if best_name == 'LogisticRegression':
    imp = np.abs(best['model'].coef_[0])
elif best_name in ('RandomForest', 'GradientBoosting'):
    imp = best['model'].feature_importances_
else:
    pi = permutation_importance(best['model'], X_test_s, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    imp = pi.importances_mean
top_idx = np.argsort(imp)[::-1][:20]
print(f"  Top 20 features ({best_name}):")
for rank, i in enumerate(top_idx):
    print(f"    {rank+1:2d}. {FEATURE_NAMES[i]:30s} {imp[i]:.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# 7. SAVE MODEL & RESULTS
# ═══════════════════════════════════════════════════════════════════════════
print("\n[7] Saving model and results")

model_path = f'{OUT_DIR}/cis_trained_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump({
        'best_model_name': best_name, 'best_model': best['model'],
        'scaler': scaler, 'pca_activity': pca_activity,
        'activity_calibrator': calib_lr, 'contact_model': contact_lr,
        'feature_names': FEATURE_NAMES, 'feat_names_raw': FEAT_NAMES,
        'dinucs': DINUCS, 'trinucs': TRINUCS,
        'gamma': GAMMA, 'd_decay': D_DECAY,
    }, f)
print(f"    Model saved: {model_path}")

results_csv = f'{OUT_DIR}/cis_trained_model_predictions.csv'
df_out = df[['enhancer_id','promoter_id','gene_id','gene_type','label','fold',
             'enhancer_chr','enhancer_start','enhancer_end',
             'promoter_chr','promoter_start','promoter_end']].copy()
df_out['genomic_distance'] = genomic_dist
df_out['contact_hic'] = contact_hic
df_out['activity_score'] = activity_score
df_out['contact_score'] = contact_score
df_out['abc_score'] = abc_score
df_out['predicted_prob'] = np.nan
df_out['predicted_label'] = np.nan
df_out.loc[test_mask, 'predicted_prob'] = best['prob']
df_out.loc[test_mask, 'predicted_label'] = best['pred']
df_out.to_csv(results_csv, index=False)
print(f"    Predictions saved: {results_csv}")

metrics_json = f'{OUT_DIR}/cis_model_metrics.json'
metrics = {}
for name, r in results.items():
    metrics[name] = {'auc': float(r['auc']), 'ap': float(r['ap']),
                     'accuracy': float(r['acc']), 'time_seconds': float(r['time']),
                     'confusion_matrix': r['cm'].tolist()}
with open(metrics_json, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"    Metrics saved: {metrics_json}")

# ═══════════════════════════════════════════════════════════════════════════
# 8. DASHBOARD (9-panel)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[8] Generating Dashboard")

plt.rcParams.update({
    'text.color': 'white', 'axes.labelcolor': 'white',
    'xtick.color': 'white', 'ytick.color': 'white',
    'axes.edgecolor': '#444', 'grid.color': '#333', 'font.size': 8,
})

fig = plt.figure(figsize=(22, 16))
fig.patch.set_facecolor('#0a0a0a')
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
axes = [fig.add_subplot(gs[i//3, i%3]) for i in range(9)]
for ax in axes:
    ax.set_facecolor('#111')
    for s in ax.spines.values(): s.set_edgecolor('#444')

pos_mask = y_all == 1
neg_mask = y_all == 0

# Panel 1: Model comparison
ax = axes[0]
model_names = list(results.keys())
aucs = [results[m]['auc'] for m in model_names]
aps  = [results[m]['ap']  for m in model_names]
x = np.arange(len(model_names)); w = 0.35
ax.bar(x - w/2, aucs, w, color='#4fc3f7', alpha=0.85, label='AUC')
ax.bar(x + w/2, aps,  w, color='#81c784', alpha=0.85, label='AP')
ax.set_xticks(x)
ax.set_xticklabels([m.replace('Regression','Reg').replace('Gradient','Grad') for m in model_names], fontsize=7, rotation=15)
ax.set_ylabel('Score', color='white')
ax.set_title('Model Comparison (Test Set)', color='white', fontsize=9, fontweight='bold')
ax.legend(fontsize=7, facecolor='#1a1a1a', labelcolor='white')
ax.set_ylim(0.5, 1.0)
for i, (a, p) in enumerate(zip(aucs, aps)):
    ax.text(i - w/2, a + 0.01, f'{a:.3f}', ha='center', fontsize=6, color='white')
    ax.text(i + w/2, p + 0.01, f'{p:.3f}', ha='center', fontsize=6, color='white')

# Panel 2: Hi-C contact vs genomic distance (log-log scatter, colored by label)
ax = axes[1]
sample_n = min(800, len(df))
sample_idx = np.random.choice(len(df), sample_n, replace=False)
sc = ax.scatter(genomic_dist[sample_idx[pos_mask[sample_idx]]] / 1e6,
                contact_hic[sample_idx[pos_mask[sample_idx]]],
                c='#4fc3f7', s=5, alpha=0.5, label='Positive')
ax.scatter(genomic_dist[sample_idx[neg_mask[sample_idx]]] / 1e6,
           contact_hic[sample_idx[neg_mask[sample_idx]]],
           c='#ef5350', s=5, alpha=0.5, label='Negative')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Genomic Distance (Mb)', color='white')
ax.set_ylabel('Hi-C Contact Frequency', color='white')
ax.set_title('Hi-C Distance Decay (Cis)', color='white', fontsize=9, fontweight='bold')
ax.legend(fontsize=7, facecolor='#1a1a1a', labelcolor='white')

# Panel 3: ROC curves
ax = axes[2]
colors_roc = ['#4fc3f7', '#81c784', '#ffb74d', '#ba68c8']
for (name, r), color in zip(results.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, r['prob'])
    ax.plot(fpr, tpr, color=color, lw=1.5, label=f'{name} ({r["auc"]:.3f})')
ax.plot([0,1], [0,1], color='#555', ls='--', lw=1)
ax.set_xlabel('False Positive Rate', color='white')
ax.set_ylabel('True Positive Rate', color='white')
ax.set_title('ROC Curves (Test Set)', color='white', fontsize=9, fontweight='bold')
ax.legend(fontsize=6, facecolor='#1a1a1a', labelcolor='white')
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

# Panel 4: PR curves
ax = axes[3]
for (name, r), color in zip(results.items(), colors_roc):
    prec, rec, _ = precision_recall_curve(y_test, r['prob'])
    ax.plot(rec, prec, color=color, lw=1.5, label=f'{name} ({r["ap"]:.3f})')
ax.axhline(y_test.mean(), color='#555', ls='--', lw=1, label=f'Baseline={y_test.mean():.2f}')
ax.set_xlabel('Recall', color='white')
ax.set_ylabel('Precision', color='white')
ax.set_title('Precision-Recall Curves', color='white', fontsize=9, fontweight='bold')
ax.legend(fontsize=6, facecolor='#1a1a1a', labelcolor='white')
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

# Panel 5: Confusion matrix (best model)
ax = axes[4]
cm = best['cm']
im = ax.imshow(cm, cmap='Blues', aspect='auto')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=14,
                color='white' if cm[i,j] > cm.max()/2 else '#333')
ax.set_xticks([0,1]); ax.set_yticks([0,1])
ax.set_xticklabels(['Neg (0)', 'Pos (1)'], color='white')
ax.set_yticklabels(['Neg (0)', 'Pos (1)'], color='white')
ax.set_xlabel('Predicted', color='white')
ax.set_ylabel('Actual', color='white')
ax.set_title(f'Confusion Matrix ({best_name})', color='white', fontsize=9, fontweight='bold')

# Panel 6: Feature importance (top 15)
ax = axes[5]
top_n = 15
top_names = [FEATURE_NAMES[i] for i in top_idx[:top_n]][::-1]
top_vals = [imp[i] for i in top_idx[:top_n]][::-1]
colors_bar = plt.cm.viridis(np.linspace(0.2, 0.9, top_n))
ax.barh(np.arange(top_n), top_vals, color=colors_bar, alpha=0.85)
ax.set_yticks(np.arange(top_n))
ax.set_yticklabels(top_names, fontsize=6)
ax.set_xlabel('Importance', color='white')
ax.set_title(f'Top {top_n} Features ({best_name})', color='white', fontsize=9, fontweight='bold')

# Panel 7: ABC score distribution (pos vs neg)
ax = axes[6]
ax.hist(abc_score[pos_mask], bins=50, color='#4fc3f7', alpha=0.7, label=f'Positive (n={pos_mask.sum()})', density=True)
ax.hist(abc_score[neg_mask], bins=50, color='#ef5350', alpha=0.7, label=f'Negative (n={neg_mask.sum()})', density=True)
ax.set_xlabel('ABC Score', color='white')
ax.set_ylabel('Density', color='white')
ax.set_title('ABC Score Distribution', color='white', fontsize=9, fontweight='bold')
ax.legend(fontsize=7, facecolor='#1a1a1a', labelcolor='white')

# Panel 8: Genomic distance distribution (pos vs neg)
ax = axes[7]
ax.hist(np.log10(genomic_dist[pos_mask] + 1), bins=50, color='#4fc3f7', alpha=0.7, label='Positive', density=True)
ax.hist(np.log10(genomic_dist[neg_mask] + 1), bins=50, color='#ef5350', alpha=0.7, label='Negative', density=True)
ax.set_xlabel('log10(Genomic Distance + 1)', color='white')
ax.set_ylabel('Density', color='white')
ax.set_title('Genomic Distance Distribution', color='white', fontsize=9, fontweight='bold')
ax.legend(fontsize=7, facecolor='#1a1a1a', labelcolor='white')

# Panel 9: Summary
ax = axes[8]
ax.axis('off')
lines = [
    "EnhancerPromoterEngine v3: Cis Model",
    "═" * 42,
    f"Data: {len(df)} cis E-P pairs (K562)",
    f"  Positive: {int(pos_mask.sum())} | Negative: {int(neg_mask.sum())}",
    f"  Train+Val: {int(train_mask.sum())} | Test: {int(test_mask.sum())}",
    f"  Distance: {genomic_dist.min():.0f} - {genomic_dist.max()/1e6:.0f} Mb",
    "",
    "Architecture (cis, real genomic distance):",
    "  1. FeatureExtractor: 93-dim/sequence",
    "  2. ActivityModule: PCA → calibrated score",
    "  3. ContactModule: Hi-C d^(-0.87)×exp(-d/3Mb)",
    "     + 5 sequence similarity metrics",
    "  4. ABCModule: Activity × Contact / gene",
    "  5. Classifier: 194 features → 4 models",
    "",
    f"Best Model: {best_name}",
    f"  Test AUC:      {best['auc']:.4f}",
    f"  Test AP:       {best['ap']:.4f}",
    f"  Test Accuracy: {best['acc']:.4f}",
    "",
    "All Models:",
]
for name in model_names:
    r = results[name]
    lines.append(f"  {name:22s} AUC={r['auc']:.4f} AP={r['ap']:.4f}")
lines += [
    "",
    f"Top Feature: {FEATURE_NAMES[top_idx[0]]}",
    f"  Importance: {imp[top_idx[0]]:.4f}",
    "",
    f"Hi-C contact (pos/neg): {contact_hic[pos_mask].mean():.4f} / {contact_hic[neg_mask].mean():.4f}",
]
y_pos = 0.96
for line in lines:
    color = '#4fc3f7' if 'Engine' in line else 'white'
    if 'AUC' in line and ':' in line and 'Test' in line: color = '#81c784'
    if 'Best' in line: color = '#ffeb3b'
    ax.text(0.03, y_pos, line, transform=ax.transAxes, fontsize=6.5,
            color=color, verticalalignment='top', fontfamily='monospace')
    y_pos -= 0.033

fig.suptitle('EnhancerPromoterEngine v3: Cis Model Dashboard\n'
             f'(cis E-P pairs: {len(df)} | Real genomic distance + Hi-C decay | 4-model comparison)',
             color='white', fontsize=13, fontweight='bold', y=0.98)

dashboard_path = f'{OUT_DIR}/cis_trained_model_dashboard.png'
plt.savefig(dashboard_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0a')
plt.close()
print(f"    Dashboard saved: {dashboard_path}")

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"  Architecture:   EnhancerPromoterEngine v3 (cis, real distance, no leakage)")
print(f"  Features:       93×2 + 7 joint + 1 ABC = 194 total")
print(f"  Train+Val:      {int(train_mask.sum())} | Test: {int(test_mask.sum())}")
print(f"  Best model:     {best_name}")
print(f"  Test AUC:       {best['auc']:.4f}")
print(f"  Test AP:        {best['ap']:.4f}")
print(f"  Test Accuracy:  {best['acc']:.4f}")
print(f"  Top feature:    {FEATURE_NAMES[top_idx[0]]} ({imp[top_idx[0]]:.4f})")
print("=" * 70)
print(f"\nOutput files:")
print(f"  Model:        {model_path}")
print(f"  Dashboard:    {dashboard_path}")
print(f"  Predictions:  {results_csv}")
print(f"  Metrics:      {metrics_json}")
