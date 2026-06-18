# 基于 BQ Corpus 文本匹配数据集试验不同方法效果报告

> **项目名称**：中文语义文本匹配多方法对比实验  
> **数据集**：BQ Corpus（银行/金融问句匹配）  
> **实验环境**：联想小新 · 纯核显 · PyTorch CPU  
> **预训练模型**：`bert-base-chinese` / `Qwen2-0.5B-Instruct`  
> **报告日期**：2026 年 6 月  

---

## 目录

1. [核心结论（必读）](#一核心结论必读)
2. [实验背景与数据](#二实验背景与数据)
3. [实验设计与环境](#三实验设计与环境)
4. [五种方法配置与结果](#四五种方法配置与结果)
5. [综合对比](#五综合对比)
6. [Bad Case 深度分析（BQ 实测）](#六bad-case-深度分析bq-实测)
7. [讨论与局限](#七讨论与局限)
8. [结论与优化建议](#八结论与优化建议)
9. [附录：复现命令](#九附录复现命令)

---

## 一、核心结论（必读）

### 1.1 一句话总结

在 **BQ Corpus validation（8,620 条）** 上，**BiEncoder + CosineEmbeddingLoss** 综合最优（**Acc/F1 = 0.7749**）；对错误样本的分析表明，模型主要失误于 **「同主题不同意图」** 与 **「跨渠道/换说法同义句」** 两类边界样本。

### 1.2 关键指标速览

| 方法 | Accuracy | F1 (weighted) | 评估规模 | 排名 |
|:----:|:--------:|:-------------:|:--------:|:----:|
| **BiEncoder + Cosine** | **0.7749** | **0.7749** | 8,620 全量 | 🥇 |
| BiEncoder + Triplet | 0.7557 | 0.7556 | 8,620 全量 | 🥈 |
| CrossEncoder | 0.7352 | 0.7350 | 8,620 全量 | 🥉 |
| Qwen SFT (LoRA) | 0.7400 | 0.7334 | 100 条 | — |
| DeepSeek zero-shot | 0.7000 | — | 50 条 | — |

> **AUC（BiEncoder Cosine）= 0.8513**，最优分类阈值 **0.71**。

### 1.3 Bad Case 核心发现（`analyze_badcases.py` 实测）

| 指标 | 数值 |
|------|------|
| 整体准确率 | **0.7749** |
| 错误总数 | **1,940 / 8,620（22.5%）** |
| FP 假阳性 | **994 条**（占错误 51.2%） |
| FN 假阴性 | **946 条**（占错误 48.8%） |
| FP 高置信度错误 | 557 条（score 距阈值 > 0.15） |
| FN 高置信度错误 | 521 条 |

**与 AFQMC 实验的差异**：BQ 上 FP/FN **几乎各半**；FP 的字符 Jaccard 均值仅 **0.230**（并非单纯「词汇重叠高」导致），更多来自 **微粒贷/电话确认/还款** 等同领域模板的 **语义边界模糊**。

---

## 二、实验背景与数据

### 2.1 任务定义

给定两个中文句子，判断是否为 **语义等价** 的金融客服问句：

- `label = 1`：相似（同义问法）
- `label = 0`：不相似（话题相关但意图不同）

**应用场景**：智能客服 FAQ 匹配、问句去重、RAG 语义检索。

### 2.2 BQ Corpus 数据规模

| 划分 | 样本数 | 正样本 | 负样本 | 正负比 |
|:----:|:------:|:------:|:------:|:------:|
| train | 68,960 | 34,438 | 34,522 | **≈1:1** |
| **validation** | **8,620** | 4,329 | 4,291 | **≈1:1** |
| test | 8,620 | 4,382 | 4,238 | ≈1:1 |

**数据特点**：

- 领域：微粒贷、还款、额度、审核、电话确认等 **银行客服口语**
- 句子极短（均值 **13.9 字**），`max_length=64` 覆盖 **99.9%**
- **无明显 length bias**（正负样本长度差接近）
- 与 AFQMC 不同：BQ **类别均衡**，不易出现「全预测负类」退化

**数据示例**：

```json
{"sentence1": "存款有保障吗", "sentence2": "不知道安全吗", "label": 1}
{"sentence1": "利息怎么计算，哪一天计起", "sentence2": "比如今天借了一万分10个月...", "label": 0}
```

---

## 三、实验设计与环境

### 3.1 硬件环境

| 项目 | 配置 |
|------|------|
| 设备 | 联想小新，**纯核显** |
| 计算 | **CPU only**（`cuda: False`） |
| 影响 | 训练慢但可完成；CrossEncoder 单 batch 可达 60～90 s |

### 3.2 实验流水线

```
explore_data.py → train_biencoder (cosine/triplet) → train_crossencoder
       ↓
evaluate.py / compare_methods.py → analyze_badcases.py
       ↓
llm_compare.py (DeepSeek) → train_sft.py → evaluate_sft.py
```

### 3.3 统一评估口径

| 方案 | 预测方式 | 主要指标 |
|------|----------|----------|
| BiEncoder | 余弦相似度 + 阈值搜索（101 档） | Acc, F1, AUC |
| CrossEncoder | argmax(logits) | Acc, F1 |
| DeepSeek | Prompt → 是/否 | Acc, F1(正例) |
| Qwen SFT | 生成【相似】/【不相似】 | Acc, F1 |

> BERT 三路在 **validation 全量 8,620 条** 上评估；LLM 因成本采用子采样（50/100 条）。

---

## 四、五种方法配置与结果

### 4.1 BiEncoder + CosineEmbeddingLoss ⭐ 最优

| 配置项 | 值 |
|--------|-----|
| 架构 | Sentence-BERT，mean pooling，**4 层 BERT（45.6M）** |
| 训练样本 | **15,000** / 68,960 |
| epochs / batch | **2** / **16** |
| max_length | 64 |
| 最优阈值 | **0.71** |

**训练曲线**：

| Epoch | train_loss | val_acc | val_f1 | 耗时 |
|:-----:|:----------:|:-------:|:------:|:----:|
| 1 | 0.2620 | 0.7666 | 0.7659 | ≈93 min |
| **2** | **0.2100** | **0.7749** | **0.7749** | ≈79 min |

**最终指标**：Accuracy **0.7749** · F1 **0.7749** · AUC **0.8513**

---

### 4.2 BiEncoder + TripletLoss

| 配置项 | 值 |
|--------|-----|
| 训练样本 | 15,000 |
| 最优阈值 | 0.67 |

**最终指标**：Accuracy **0.7557** · F1 **0.7556** · AUC **0.8320**

**vs Cosine**：ΔAcc **-0.0192** · ΔF1 **-0.0193** → **Cosine 更优**

---

### 4.3 CrossEncoder + CrossEntropyLoss

| 配置项 | 值 |
|--------|-----|
| 训练样本 | **5,000**（CPU 时间限制） |
| epochs | **1** |
| batch / max_length | 12 / 96 |

**最终指标**：Accuracy **0.7352** · F1 **0.7350**

```
              precision    recall    f1-score    support
Not similar      0.74       0.71       0.73       4291
    Similar      0.73       0.76       0.74       4329
```

训练集 acc 仅 0.66 → **训练不充分**，增加数据/epoch 后有提升空间。

---

### 4.4 DeepSeek API zero-shot

| 配置 | 值 |
|------|-----|
| 模型 | deepseek-chat |
| 样本 | **50 条** |

| 指标 | 值 |
|------|-----|
| Accuracy | 0.7000 |
| 正例 Recall | **0.1250**（极低） |
| **F1(正例)** | **0.2105** |

**问题**：大量 FN，对短句/省略式问法过于保守，**不适合单独用于金融匹配生产**。

---

### 4.5 Qwen2-0.5B LoRA SFT

| 配置 | 值 |
|------|-----|
| LoRA 可训练参数 | **0.22%**（1.08M / 495M） |
| 训练 | 1,500 条平衡采样 · 1 epoch · **≈44 min** |
| 评估 | **100 条** |

| 指标 | 值 |
|------|-----|
| Accuracy | 0.7400 |
| F1 (weighted) | 0.7334 |
| **F1(正例)** | **0.7903** |
| parse_fail | **0** |

**亮点**：正例 F1 优于 DeepSeek zero-shot；**不足**：全量 Acc 略低于 BiEncoder Cosine。

---

## 五、综合对比

### 5.1 全方法结果表

| 方法 | Accuracy | F1(w) | F1(正) | AUC/阈值 | 训练量 | 评估量 |
|------|:--------:|:-----:|:------:|:--------:|:------:|:------:|
| **BiEncoder Cosine** | **0.7749** | **0.7749** | ≈0.775 | 0.851 / 0.71 | 15K | **8620** |
| BiEncoder Triplet | 0.7557 | 0.7556 | ≈0.756 | 0.832 / 0.67 | 15K | 8620 |
| CrossEncoder | 0.7352 | 0.7350 | 0.74 | argmax | 5K | 8620 |
| Qwen SFT | 0.7400 | 0.7334 | **0.7903** | — | 1.5K | 100 |
| DeepSeek | 0.7000 | — | 0.2105 | — | 0 | 50 |

### 5.2 方法选型建议

| 场景 | 推荐方案 | 理由 |
|------|----------|------|
| FAQ 大规模召回 | **BiEncoder Cosine** | 最高 F1 + 可向量化 |
| 精排 Top-K | **CrossEncoder** | 全层交互，精度潜力高 |
| 本地离线、无 API | **Qwen SFT** | 少量数据即可，正例 F1 好 |
| 快速原型 | DeepSeek zero-shot | 免训练，但 **正例召回差** |
| 生产级联 | BiEncoder 召回 → CrossEncoder 精排 | 速度 + 精度兼顾 |

### 5.3 可视化产出

| 文件 | 内容 |
|------|------|
| `biencoder_validation_sim_dist.png` | 相似度分布 |
| `method_comparison_bar.png` | 三方法柱状对比 |
| `biencoder_sim_distributions.png` | Cosine vs Triplet 分布 |
| **`biencoder_badcase_dist.png`** | **正确 vs 错误 分数分布** |
| `label_distribution.png` 等 | 数据探索 |

---

## 六、Bad Case 深度分析（BQ 实测）

> **分析脚本**：`python analyze_badcases.py --batch_size 16 --n_cases 5`  
> **分析模型**：`biencoder_cosine_best.pt`（Cosine BiEncoder，阈值 **0.71**）  
> **分析数据**：`data/bq_corpus/validation.jsonl`，**8,620 条全量**

### 6.1 错误总览

```
整体准确率：0.7749
错误总数：  1,940 条（占 22.5%）
正确预测：  6,680 条（占 77.5%）
```

| 错误类型 | 数量 | 占错误比例 | 含义 |
|----------|:----:|:----------:|------|
| **FP 假阳性** | **994** | 51.2% | 预测相似，实际不相似 |
| **FN 假阴性** | **946** | 48.8% | 预测不相似，实际相似 |

| 子类型 | FP | FN |
|--------|:--:|:--:|
| **高置信度错误**（距阈值 > 0.15） | 557 | 521 |
| **临界错误**（距阈值 ≤ 0.15） | 437 | 425 |

**解读**：

- FP 略多于 FN，说明模型在 BQ 上 **略偏「宽松」**（更容易判相似）
- 高置信度错误合计 **1,078 条**，占全部错误的 **55.6%** → 并非全是「阈值边界」问题，存在 **系统性语义混淆**
- 临界错误各约 430 条 → 约 **22% 的错误** 可通过阈值微调部分缓解

### 6.2 语言特征统计（BQ 特有发现）

| 特征 | FP（994 条） | FN（946 条） | 对比 |
|------|:------------:|:------------:|------|
| 句子长度差均值 | 5.6 | 6.8 | 接近 |
| s1 长度均值 | 11.6 | 12.5 | 短句为主 |
| **字符 Jaccard 均值** | **0.230** | **0.203** | **均较低** |

> **重要发现（与 AFQMC 实验不同）**  
> 在 AFQMC 上 FP 的 Jaccard 往往 **>0.5**（词汇高度重叠）；  
> 在 **BQ Corpus** 上 FP/FN 的 Jaccard 均约 **0.20**，说明 BQ 的错误 **不主要由表面词重叠驱动**，而是 **同领域模板下的意图细微差别** 与 **换说法/跨渠道表达** 导致。

### 6.3 FP 典型案例（预测相似 · 实际不相似）

#### 高置信度 FP（score ≈ 0.99）— 「电话/确认」主题过度泛化

| score | 句子 A | 句子 B |
|:-----:|--------|--------|
| 0.994 | 前面打电话没接到 | 没接到电话 |
| 0.994 | 可以在次拨打电话吗 | 可以主动打电话过去吗 |
| 0.993 | 什么情况，我没接到电话 | 3次电话都没有接到怎么办 |
| 0.993 | 打我电话没接到，刚在忙 | 咋我借钱之后电话确认，到现在就一个电话还没接到 |
| 0.993 | 自己打电话确认可以吗 | 现在可以打电话过来给我吗？ |

**成因分析**：两句均围绕 **「电话 / 确认 / 没接到」**，向量空间中距离极近，但用户意图可能不同（陈述事实 vs 询问规则 vs 投诉时效）。模型学到了 **领域关键词共现 ≈ 相似**，未能区分 **细粒度意图**。

#### 临界 FP（score 0.75～0.86）— 相关话题、不同问法

| score | 句子 A | 句子 B |
|:-----:|--------|--------|
| 0.764 | hello请问发起借款后多长时间可到 | QQ微粒代体现要多久 |
| 0.860 | 为什么绑定银行。说对方银行处理失败 | 为什么总是失败 |
| 0.816 | 我今天有一笔款到期，能否延迟两天 | 延迟一天还款 |
| 0.772 | 怎样关闭微粒贷怎 | 在我操作微信贷款的时候，预留的手机号已经停止使用了。 |

**成因分析**：共享 **借款/还款/绑定/关闭** 等主题，但一问一答的 **具体诉求不同**；处于阈值 **0.71 附近**，属于 **可经精排或 CrossEncoder 修正** 的样本。

### 6.4 FN 典型案例（预测不相似 · 实际相似）

#### 高置信度 FN（score 为负或极低）— 跨渠道 / 换说法

| score | 句子 A | 句子 B |
|:-----:|--------|--------|
| -0.046 | QQ现金贷我都有名额 | QQ钱包看不到微粒贷呢 |
| -0.045 | 单笔最多贷几万一天最多货几万 | 一次性代最高额度会通过吗 |
| -0.038 | 你好，我申请了贷款，什么时候能下来 | 为何还没有入账。 |
| -0.030 | QQ和微信是一个银行借钱口吗 | 为什么微信上面有微粒贷图标，怎么QQ上面没显示了？ |
| -0.027 | 可以给我资格么？ | /微笑，我怎么还是没有微粒贷 |

**成因分析**：

- **QQ vs 微信 vs 微粒贷** 等渠道差异，表面词汇重叠低
- 错别字、口语省略（「货几万」「代最高」）
- 模型 **未建立「同义表达」映射**，向量距离过远

#### 临界 FN（score 0.56～0.70）— 接近阈值但未过线

| score | 句子 A | 句子 B |
|:-----:|--------|--------|
| 0.686 | 可以先第一个月的利息，第二个月一次性还清么？… | 微粒贷怎么计息 |
| 0.701 | 为什么我绑不上银行卡 | 总是绑定失败 |
| 0.702 | 我的还款日期 | 我几号开始还款 |
| 0.660 | 能不能帮我关闭微粒贷 | 我要退出微粒贷 |
| 0.563 | 银行确定 | 银行电话确认 |

**成因分析**：语义 **高度相近**，score 已在 **0.56～0.70**，略低于阈值 **0.71**；降低阈值可减少 FN，但可能增加 FP → 需 **PR 曲线权衡** 或 **CrossEncoder 精排**。

### 6.5 错误分布图

**输出文件**：`outputs/figures/biencoder_badcase_dist.png`

- 左图：正/负样本分数分布 + 阈值线  
- 右图：预测正确 vs 预测错误的分数分布  

**观察**：正确样本集中在阈值两侧；错误样本在 **阈值附近重叠区** 与 **高置信度远离区** 均存在，印证 6.1 节「临界 + 系统性错误」并存。

### 6.6 基于 BQ Bad Case 的优化方向

| 优先级 | 方向 | 针对问题 | 具体做法 |
|:------:|------|----------|----------|
| ⭐⭐⭐ | **BiEncoder + CrossEncoder 级联** | FP 高置信「同主题不同意图」 | 召回 Top-50 → CrossEncoder 精排 Top-1 |
| ⭐⭐⭐ | **难负样本挖掘** | FP（相似度高但 label=0） | 用当前模型挖掘 hard negatives 加入训练 |
| ⭐⭐ | **同义句数据增强** | FN（换说法/跨渠道） | LLM 改写正样本；补充 QQ/微信/微粒贷 同义对 |
| ⭐⭐ | **全量训练 + 更深 BERT** | 系统性错误 | 69K 全量 · 12 层 · 更多 epoch |
| ⭐ | **阈值校准** | 临界错误 (~430×2) | 按业务 Precision/Recall 需求调整 0.71 |
| ⭐ | **领域预训练模型** | 金融术语 | MacBERT / 金融语料继续预训练 |

---

## 七、讨论与局限

### 7.1 表示型 vs 交互型

| | BiEncoder | CrossEncoder |
|---|-----------|--------------|
| 本实验 F1 | **0.775** | 0.735 |
| 速度 | 快（可预计算向量） | 慢（每对重算） |
| Bad Case | FP 多来自同主题混淆 | 未单独分析，可作精排补充 |

### 7.2 判别式 vs 生成式

- **BERT Cosine**：全量最优，适合 **主匹配引擎 + 向量库**  
- **SFT**：正例 F1 高，适合 **本地部署、格式稳定输出**  
- **DeepSeek zero-shot**：Acc 0.70 但 **正例 F1 0.21**，仅适合 **快速验证**

### 7.3 实验局限

1. LLM 评估为 **子采样**（50/100 条），与 BERT 全量对比有误差  
2. 全部 **CPU** 训练，CrossEncoder 仅用 **5K×1 epoch**  
3. Bad Case 分析目前针对 **BiEncoder Cosine**；CrossEncoder / SFT 可扩展  
4. 最终指标以 **validation** 为准，未报告 test 集  

### 7.4 运行耗时参考（联想小新 CPU）

| 阶段 | 耗时（约） |
|------|:----------:|
| BiEncoder Cosine 训练（15K×2） | 3 h |
| CrossEncoder 训练（5K×1） | 24 min |
| compare_methods（8620×3） | 1～2.5 h |
| **analyze_badcases（8620）** | **~30～60 min** |
| SFT 训练（1500×1） | 44 min |

---

## 八、结论与优化建议

### 8.1 主要结论

1. **BiEncoder + CosineEmbeddingLoss** 在 BQ Corpus 上综合最优（**Acc/F1 = 0.7749，AUC = 0.851**）。  
2. **TripletLoss** 低于 Cosine（ΔF1 ≈ -0.019），直接标签监督更稳定。  
3. **CrossEncoder** F1 = 0.735，适合作为 **二阶段精排**，弥补 BiEncoder 的 FP 问题。  
4. **Bad Case 分析** 显示：994 FP + 946 FN，错误 **近半为高置信度**，BQ 上需关注 **同主题意图混淆** 与 **跨渠道同义表达**，而非单纯词汇重叠。  
5. **DeepSeek zero-shot** 正例 F1 仅 **0.21**，不能替代微调模型；**Qwen SFT** 用 1,500 条达 **F1(正例) 0.79**，证明小模型领域微调价值。  

### 8.2 后续工作

- [ ] 全量 69K 训练 BiEncoder  
- [ ] 实施 BiEncoder → CrossEncoder 级联并在 Bad Case 集上复测  
- [ ] 针对 FP「电话/确认」类与 FN「QQ/微信渠道」类构造专项训练样本  
- [ ] SFT / DeepSeek 在 **全量 8620** 上评估以便公平对比  

---

## 九、附录：复现命令

### 数据与探索

```powershell
cd src
python explore_data.py
```

### BERT 训练

```powershell
python train_biencoder.py --loss cosine --max_train_samples 15000 --epochs 2 --batch_size 16
python train_biencoder.py --loss triplet  --max_train_samples 15000 --epochs 2 --batch_size 16
python train_crossencoder.py --max_train_samples 5000 --epochs 1 --batch_size 12 --max_length 96
```

### 评估、对比与 Bad Case

```powershell
python compare_methods.py --batch_size 16
python evaluate.py --model_type biencoder --ckpt ../outputs/checkpoints/biencoder_cosine_best.pt --batch_size 16

# BQ Bad Case 分析（本报告第六节数据来源）
python analyze_badcases.py --batch_size 16 --n_cases 5
```

### LLM 路线

```powershell
cd ..\src_llm
$env:DEEPSEEK_API_KEY="sk-xxx"
python llm_compare.py --num_samples 50
python train_sft.py --num_train 1500 --epochs 1 --batch_size 1 --grad_accum 16
python evaluate_sft.py --num_samples 100
```

### 关键输出

```
outputs/checkpoints/     biencoder_cosine_best.pt 等
outputs/logs/            method_comparison.json, sft_results.json 等
outputs/figures/         biencoder_badcase_dist.png 等
```

---

**报告完**

*数值来源：`outputs/logs/`、`outputs/checkpoints/`、`analyze_badcases.py` 于 BQ validation 全量 8,620 条的终端实测输出。*
