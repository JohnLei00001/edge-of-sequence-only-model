# ENdb — 增强子-启动子互作序列数据集

从公开增强子数据库（**ENdb** / **EP_ATLAS**）构建面向 **sequence-only 模型**（纯序列预测增强子-启动子互作）的训练数据集。项目以提取增强子/启动子的**基因组序列**为核心，并为建模产出**平衡的 cis/trans 正负样本**与按增强子无泄漏的 train/val/test 切分。

> 参考基因组为 **hg19**，序列坐标遵循 **1-based 全闭区间**约定。

## 项目结构

```
.
├── EP_ATLAS/                    # 当前主数据集（K562 平衡 cis/trans 集）
│   ├── K562_Genos_balanced_cis_trans_2526_pairs.csv   # 正样本 2526 对
│   ├── K562_download.txt        # 完整互作全集（91,310 行），作负样本阳性过滤器
│   ├── negative_sampling_report.md  # 负样本构建与质量校验报告
│   ├── cis/                     # cis（同染色体）子集，含标注+序列（不入库）
│   ├── trans/                   # trans（跨染色体）子集，含标注+序列（不入库）
│   └── sequences/               # 从 K562 CSV 提取的唯一序列 FASTA（不入库）
├── scripts/                     # 全部构建脚本
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
- 大数据 / 生成产物（参考基因组、archive、FASTA、cis/trans 序列目录）由 `.gitignore` 排除，不入库。
