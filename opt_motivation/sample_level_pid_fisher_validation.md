# Sample-Level 验证：PID 不 sufficient，需要 Fisher

本文档记录如何在标准 `IMDb` 设定下，做一个尽量 `fair / solid / 不弱化 PID` 的 sample-level 实验，来验证下面这个动机：

- `PID` 能描述样本中存在什么样的跨模态信息结构；
- 但 `PID` 不能充分解释 sample-level 的 learnability / correctness；
- 因此，在 `PID` 之外，还需要一个 optimization-side 的量来分析样本是否“学得出来”，这里使用 `Fisher`。

## 1. 核心原则

这个实验的立足点不是证明 `Fisher > PID`，而是证明：

- `PID is necessary but not sufficient.`
- 在 `PID` 已经尽可能强的前提下，sample-level outcome 仍然存在残差异质性；
- `Fisher` 能解释这部分 residual heterogeneity 的一部分。

因此实验必须满足以下公平性要求：

1. `PID` 不用 oracle，不用 `conf_joint/conf_i` 这类从最终预测直接构造的量。
2. `PID` 用从 data 估计得到的 `LSMI`，并使用完整 profile，而不是只用 `synergy`。
3. 模型用常规 `IMDb` joint training，不引入 staged training、curriculum、dropout 干预。
4. 分析重点放在 `PID-conditioned` 的 sample-level residual，而不是只看 dataset-level 平均 AUC。

## 2. 数据与模型

### 数据

- 数据集：`IMDb`
- 路径：`/data/zhenshuo/datasets/multimodal_imdb.hdf5`

### 标准训练设定

- 不做 text dropout
- 不做 low-resource 特殊裁剪
- 不做人为分阶段训练
- 使用现有 `opt_motivation` 标准 joint/unimodal pipeline

对应代码：

- 主训练入口：`opt_motivation/run_decoupling_experiment.py`
- 多 seed 聚合：`opt_motivation/run_imdb_multiseed.py`
- sample-level 分析：`opt_motivation/analyze_imdb_sample_level.py`

## 3. PID 与 Fisher 的定义

### PID

在 `IMDb` 上，PID 特征来自 `LSMI`，采用完整 4 维：

- `redundancy`
- `unique_left`
- `unique_right`
- `synergy`

要求：

- 不弱化 PID
- 不只取单一 `synergy`
- 不用 oracle label-conditioned confidence 替代 data-side PID

### Fisher

当前使用的 optimization-side 特征为：

- `GradCos`
- `CurvAgree`
- 以及聚合后的 `fisher_score`

它们来自 joint model 的 optimization geometry，目的不是替代 PID，而是描述当前模型下 cross-modal information 的 optimization accessibility。

## 4. 为什么不能只看 dataset-level 平均

如果只比较：

- `PID-only`
- `Fisher-only`
- `PID+Fisher`

那么结果容易被解读成“又加了几个 feature，预测更好了”，会显得比较 incremental。

因此，主实验不应只停留在平均 AUC / R² 上，而应转向下面这个问题：

- 在 `PID` 很接近的 sample 之间，为什么最终 `correctness` 仍然不同？
- `Fisher` 能否解释这部分 `PID` 无法解释的 residual difference？

## 5. 实验设计

### Step 1: 跑标准 IMDb 多 seed

目的：

- 先固定标准训练设定
- 消除单个 seed 波动
- 为后续 sample-level 聚合提供稳定输入

建议命令：

```bash
conda run -n vlm python /data/zhenshuo/multimodal/Multimodal-representation-and-attribution/opt_motivation/run_imdb_multiseed.py \
  --seeds 42 43 44 \
  --output-json /data/zhenshuo/multimodal/Multimodal-representation-and-attribution/opt_motivation/results/imdb_lsmi_fisher_multiseed_summary.json \
  --output-dir /data/zhenshuo/multimodal/Multimodal-representation-and-attribution/opt_motivation/results/multiseed_runs
```

得到的每个 seed 结果中，必须包含：

- `pid_feature_values`
- `pid_feature_names`
- `gradcos`
- `curvagree`
- `fisher_score`
- `correctness`
- `negative_loss_advantage`

### Step 2: 做 sample-level 聚合分析

目的：

- 在多个 seed 上聚合 sample-level PID / Fisher / outcome
- 估计 `PID-conditioned` 下的 Fisher 残差解释力

命令：

```bash
conda run -n vlm python /data/zhenshuo/multimodal/Multimodal-representation-and-attribution/opt_motivation/analyze_imdb_sample_level.py
```

默认输出：

- JSON：`opt_motivation/results/imdb_sample_level_analysis.json`
- 图：`opt_motivation/results/figures/imdb_sample_level_scatter.png`

## 6. 主要分析块

### A. PID-neighborhood residual analysis

思路：

1. 在完整 `LSMI PID` 空间里，对每个 sample 找 `kNN`
2. 把这些邻居视为“PID 很接近”的局部 neighborhood
3. 计算该 sample 的：
   - `correctness_residual = sample_correctness - neighborhood_mean_correctness`
4. 再检验：
   - `Fisher rank` 与 `correctness_residual` 是否正相关

这一步回答的问题是：

- 即使 `PID` 相近，sample-level correctness 是否仍然会随 `Fisher` 系统性变化？

主检验：

- `Spearman(fisher_rank, correctness_residual)`
- `top quartile Fisher rank` vs `bottom quartile Fisher rank` 的 residual contrast

### B. Local Fisher stratification

思路：

对每个 anchor sample：

1. 找它的 PID 邻域
2. 按 Fisher 把邻域样本分成高/低两组
3. 比较：
   - `high_fisher_mean_correctness - low_fisher_mean_correctness`

如果大量 anchor 都满足这个差值为正，就说明：

- 在同样的 PID 局部结构下，高 Fisher 样本更容易有更高 correctness

主检验：

- 正差值 anchor 的比例
- sign / binomial test
- Wilcoxon test

### C. PID-matched pair concordance

思路：

1. 从 PID 邻域中构造两两 pair
2. 只保留 `PID distance` 很小的 pair
3. 检验：
   - `sign(ΔFisher)` 是否与 `sign(ΔCorrectness)` 一致

主检验：

- 全部 pair 的 concordance
- 更紧的 `PID distance` 子集上的 concordance

这一步回答：

- 在几乎相同 PID 的两条样本之间，Fisher 更高的一边是否更可能 correctness 更高？

## 7. 主要 outcome 的选择

当前建议把主 outcome 收紧到：

- `correctness`

不建议把下面两项作为主结论：

- `negative_loss_advantage`
- `beats_best_unimodal`

原因：

- 当前 `IMDb` 上，这两项对 Fisher 的支持不如 `correctness` 稳定；
- 如果强行一并主打，会削弱整体论证。

## 8. 当前这版 IMDb 的推荐主结果

基于当前标准 `IMDb` 多 seed sample-level 分析，最适合作为 motivation 证据的是：

### A. PID-neighborhood residual

- `Spearman(fisher_rank, correctness_residual) = 0.112`
- `p = 2.90e-23`

同一 PID 邻域中：

- Fisher-rank 最高四分位的 `correctness residual`
- 相比最低四分位平均高 `+0.1004`
- `p = 9.86e-20`

### B. Local Fisher stratification

- 共 `7799` 个 anchor sample
- `5068` 个 anchor 的 `correctness_delta > 0`
- 去除完全相等的 anchor 后，正向比例 `71.8%`
- `binom p = 9.74e-305`
- 平均 `correctness_delta = +0.0988`

### C. PID-matched pair concordance

全部非零 correctness gap 的 PID-neighbor pairs：

- concordance `= 0.574`

更紧的 PID pair：

- top 25% tight pairs: concordance `= 0.647`
- top 10% tight pairs: concordance `= 0.681`

说明：

- 在 PID 更接近时，Fisher 与 correctness 的一致性更强；
- 这更符合“PID 不 sufficient，Fisher 解释 residual heterogeneity”的论点。

## 9. 论文里应该怎么表述

推荐表述：

- `PID` 是分析 multimodal information structure 的必要起点；
- 但在 sample level，`PID` 不是 sufficient，因为 `PID` 相近的样本仍然表现出明显不同的 correctness；
- `Fisher` 不替代 `PID`，而是解释 `PID` 无法解释的那部分 residual learnability；
- 因此，多模态分析应从 `PID-only` 扩展到 `PID-conditioned optimization geometry analysis`。

不推荐表述：

- `Fisher is better than PID`
- `Fisher replaces PID`
- `PID fails`

## 10. 这个实验为什么不算 incremental

如果只做：

- `PID-only` vs `PID+Fisher` 的平均预测比较

那确实容易显得 incremental。

但当前这套实验真正回答的是：

- 为什么 `PID` 相似的样本，最终 learning outcome 仍然不同？
- 这部分 residual heterogeneity 能否由 optimization geometry 解释？

这已经不是“再加一个 feature”，而是在 `PID` 框架内明确指出：

- `information availability != learnability`

并给出一个 sample-level、conditioned、mechanistic 的分析路径。

## 11. 最终建议

在完整论文中，这一块最适合作为：

- `motivation` 的实验支撑
- `PID 不 sufficient` 的证据
- 引入后续 Fisher-based analysis 的桥梁

它不应该被包装成整篇论文的唯一 contribution，而应该承担下面这个角色：

- 先证明现有 PID 视角的边界
- 再自然引出后续基于 Fisher 的核心观点与方法
