# ENdb — 增强子-启动子互作序列数据集与预测模型

从公开增强子数据库（**ENdb** / **EP_ATLAS**）构建面向 **sequence-only 模型**（纯序列预测增强子-启动子互作）的训练数据集，并在其上训练/评估分类模型。项目以提取增强子/启动子的**基因组序列**为核心，产出**平衡的 cis/trans 正负样本**与按增强子无泄漏的 train/val/test 切分，进而比较 ABC 特征、序列 embedding 及其混合的特征配置，并通过严格的折内降维 + 分组交叉验证得出显著性结论。

> 参考基因组为 **hg19**，序列坐标遵循 **1-based 全闭区间**约定。

## 项目结构

```
.
├── EP_ATLAS/                    # 当前主数据集（K562 平衡 cis/trans 集）
│   ├── K562_Genos_balanced_cis_trans_2526_pairs.csv   # 正样本 2526 对
│   ├── K562_download.txt        # 完整互作全集（91,310 行），作负样本阳性过滤器
│   ├── negative_sampling_report.md  # 负样本构建与质量校验报告
│   ├── cis/                     # cis 子集，含标注+序列（不入库）
│   ├── trans/                   # trans 子集，含标注+序列（不入库）
│   ├── sequences/               # 唯一序列 FASTA（不入库）
│   ├── cache/                   # 对齐后的 ABC/CO/SEP 特征矩阵缓存 + manifest
│   ├── models/                  # 训练/评估报告 JSON 与模型目录
│   ├── report/                  # 图表 + docx 报告（中/英文）
│   └── results/                 # CV/显著性/降维搜索结果与图
├── scripts/                     # 全部构建 + 训练/评估脚本
├── ABC_Score/                   # 纯 ABC 参考实现（EnhancerPromoterEngine v3）
├── references/                  # 参考基因组 / 注释（本地，不入库）
├── MAP/ genos/                  # 辅助资源（genos/api.txt 为密钥，被 gitignore）
├── archive/                     # ENdb 多物种原始数据与早期产物（不入库）
└── .gitignore
```

## 当前主数据集（K562）

- **正样本**：`K562_Genos_balanced_cis_trans_2526_pairs.csv`，2,526 对（cis 1,263 + trans 1,263），含增强子/启动子坐标、ENSG、基因、reads、P 值、方法与 `interaction_type`。
- **完整全集**：`K562_download.txt`，91,310 行 / 41,307 唯一互作对，用于过滤掉有实验证据的候选负样本。
- **负样本**：`scripts/build_negatives.py` 按正样本分布构造等量负样本，产出
  `cis/cis_pairs_labeled.tsv` 与 `trans/trans_pairs_labeled.tsv`（各 2,526 行，label 1/0），并做按增强子无泄漏的 70/15/15 切分。方法与校验见 `negative_sampling_report.md`。

## 构建流程

数据流水线整体为：**原始数据 → 清洗 → 坐标/序列提取 → 主表归总 → 负样本 → 校验**。

| 脚本 | 作用 |
|---|---|
| `scripts/clean_endb.py` | 清洗 ENdb `enhancer_main.txt`（修复拆分行、统一列名），输出规范 TSV |
| `scripts/build_tss_index.py` | 从 GENCODE / UCSC refGene 构建基因 symbol → 规范 TSS 索引 |
| `scripts/build_sequences.py` | 按增强子坐标切片 + 靶基因 TSS±窗口提取启动子序列，支持多物种多 build |
| `scripts/validate_sequences.py` | 校验序列长度、N 比例、链方向反向互补 |
| `scripts/extract_ep_atlas.py` | 从 K562 CSV 按 ID 去重提取唯一增强子/启动子序列（早期路径） |
| `scripts/build_epatlas_master.py` | 将互作关联表与序列合并为自包含主表（供 sequence-only 模型直接使用） |
| `scripts/build_cis_trans_sequences.py` | 按 `interaction_type` 拆分 cis/trans，分别提取序列，分目录存放 |
| `scripts/build_negatives.py` | 构建平衡 cis/trans 负样本 + 无泄漏切分 + 生成报告 |

### 负样本构造逻辑（cis/trans）

- **cis 负样本**：cis 池同染色体组合 → 剔除全集有证据的对 → 按正样本距离分布**分箱分层抽样**。
- **trans 负样本**：trans 池跨染色体组合 → 剔除全集有证据的对 → 随机抽样。
- 核心原则：**打乱配对 + 完整互作全集作阳性过滤**，保证负样本零实验证据；按增强子无泄漏切分。

## 模型训练与评估流水线

数据之上进一步比较 **ABC 特征**、**序列 embedding** 及其混合配置。整条流水线遵循两条防泄漏红线：**所有降维器/标准化器/calibrator 仅在训练折内 fit**；交叉验证**按增强子身份分组**（同一增强子不跨折）。

### 阶段与脚本

| 阶段 | 脚本 | 作用 |
|---|---|---|
| Embedding | `embed_sequences.py` | 批量调 Genos `dna_embedding` API，为唯一增强子/启动子生成 1024 维 embedding（并发+重试+续跑） |
| Embedding | `embed_co.py` | co-embedding：增强子+100N+启动子拼一条序列再 embed，逐对生成 1024 维向量 |
| Embedding | `test_genos_api.py` | 小批量验证 Genos API 端点/字段/维度/耗时 |
| 特征构建 | `build_pair_features.py` | 拼 2049 维 `[enh_emb \| prom_emb \| 余弦相似度]` 特征 |
| 特征构建 | `prepare_cache.py` | 对齐并缓存 ABC/CO/SEP 三类特征矩阵 + y/分组/fold，落 manifest |
| 降维/搜索 | `train_dr_grid.py`、`search_dr.py`、`search_dr_extend.py`、`supervised_dr_cv.py` | 网格搜索最优降维配置（结论 **KernelPCA-RBF, 100 维**，折内 fit） |
| 训练/CV | `train_classifiers.py`、`train_hybrid.py`、`train_hybrid_pca.py` | SVM/RF/XGB/LR/GB/MLP 二分类，折内调参防泄漏 |
| 训练/CV | `grouped_cv.py`、`coembed_cv.py`、`compare_4configs_cv.py`、`compare_8configs.py` | 分组 5 折 CV，比较多种特征配置 × {GB, RF} |
| 显著性 | `collect_all_oof.py`、`test_significance.py`、`final_eval.py` | 收集全量 OOF 预测，配对 bootstrap AUC 差 95%CI + p + 逐折 Wilcoxon |
| 最终评估 | `eval_final_cis.py`、`final_report.py` | 最优配置 4 分类器 held-out test 评估 + Dashboard + 图表 |
| 报告 | `build_complete_report.py`、`build_stage1/stage2_report(_v2).py`、`compare_configs_report.py` | 生成中/英文 docx 报告 |
| ABC 参考 | `ABC_Score/train_cis_model.py` | 纯 ABC 端到端参考实现（EnhancerPromoterEngine v3，含 93 维序列特征+活性/接触/ABC 模块） |

### 特征定义

- **ABC 特征（194 维）**：93 维序列组成特征 + Hi-C 接触衰减 `C(d)=d^-0.87·exp(-d/3Mb)` + 余弦相似度 + ABC 评分（活性×接触按基因归一），参数见 `cache/manifest.json`（gamma=0.87, D=3e6, activity_PCA=10）。
- **独立拼接 embedding（SEP，2049 维）**：`[enh_emb | prom_emb | 余弦相似度]`。
- **co-embedding（CO，1024 维）**：增强子+100N+启动子的拼接序列 embedding。

### 关键结果

- **ABC 是主导信号**（AUC≈0.69）；单独 embedding 几乎无信号（0.53–0.56，因同物种序列高度共线，余弦≈0.99）。
- **ABC + co-embedding（降维后）最佳**：AUC≈0.725，**显著优于纯 ABC（+0.034）**。
- 通过配对 bootstrap（10,000 次）与逐折 Wilcoxon 做显著性检验；详细图表与报告见 `EP_ATLAS/report/`、`EP_ATLAS/results/`。

## 依赖

- Python 3，`pyfaidx`（读取 `.fa` / `.fa.gz` 参考基因组）。
- 参考基因组需按各脚本预期的路径放置（如 `references/hg19/hg19.fa`）。

## 运行

```bash
# 例：为 K562 平衡集构建负样本并生成报告
python scripts/build_negatives.py

# 例：按 cis/trans 拆分并提取序列
python scripts/build_cis_trans_sequences.py
```

多数脚本以仓库根为基准解析输入/输出路径，直接 `python scripts/<name>.py` 即可。

## 约定

- 坐标均为 **1-based 全闭区间**；`pyfaidx` fetch 同约定。
- 启动子按基因编码链方向取 TSS-2000 ~ TSS+500（`-` 链反向互补）。
- 大数据 / 生成产物（参考基因组、archive、FASTA、cis/trans 序列目录、sequences/embeddings/features、`.npy/.npz`）由 `.gitignore` 排除，不入库。
