# TCRPPO 架构审计与下一代模型重构方案 (Audit & Redesign Proposal)

基于对当前 `TCRPPO` 代码库（特别是 `tcr_env.py`, `reward.py`, `policy.py`）的深入审查，我为你整理了这份详尽的架构改进与重构指南（`query.md`）。本文档针对你提出的痛点，从**动作空间设计**、**代理打分模型**、**负样本抗靶向设计**以及**序列天然性约束**四个维度进行了深度剖析，并给出了具体的 Next-step 方案。

---

## 1. 动作空间限制：缺失的插入（Insertion）与删除（Deletion）

### 🛑 当前缺陷审计
在 `code/tcr_env.py` 的 `_edit_sequence` 函数中，当前的动作被严格限制为**定点替换 (Substitution/Edit)**：
```python
# code/tcr_env.py
new_peptide = peptide[:position] + AMINO_ACIDS[action[1]-1] + peptide[position+1:]
```
目前的 Action Space 是 `MultiDiscrete([max_len, 20])`（选定位置并替换为20种氨基酸之一）。这种设计导致：
1. **序列长度锁定**：输入初始 TCR 长度为 $L$，输出依然是 $L$。
2. **探索空间受限**：生物学中真实的 CDR3 进化（VDJ 重排）不仅包含点突变，大量的多样性来源于核苷酸的插入和删除。固定长度直接切断了模型探索最佳 CDR3 长度回路的可能性。

### 🚀 改进方案
**将 Action Space 升级为支持 Indel 的多模态空间：**
1. **重新定义动作空间**：将其扩展为 `MultiDiscrete([max_len, 3, 20])`，其中中间的维度代表操作类型：`0=Substitute, 1=Insert, 2=Delete`。
2. **环境状态自适应 (Dynamic State Padding)**：因为 RL 的 Observation Space 需要固定维度（如 `max_len=27`），我们可以在序列尾部引入一个特殊的 `<PAD>` token。
   - **Insert**：在目标位置插入氨基酸，序列整体后移，挤掉末尾的一个 `<PAD>`。
   - **Delete**：删除目标位置的氨基酸，序列整体前移，在末尾补充一个 `<PAD>`。
3. **彻底抛弃定长框架**：更激进/现代的做法是彻底放弃基于固定长度状态的 PPO，转而采用**自回归生成 (Autoregressive Generation)** 配合 RL（类似于 RLHF 训练 LLM 的方式），基于 Transformer 的 Policy 网络一次生成一个氨基酸，遇 `<EOS>` 停止，天然支持任意长度。

---

## 2. 亲和力打分模型的准确性与鲁棒性

### 🛑 当前缺陷审计
TCRPPO 严重依赖 ERGO 作为唯一的 Reward 信号源 (`reward.py` 中的 `__get_ergo_preds`)。
1. **Reward Hacking (对抗样本漏洞)**：RL 极其擅长寻找打分模型的盲区。ERGO 是一个基于浅层 LSTM/AE 的模型，它很容易给一些毫无生物学意义但恰好激活了其特定神经元的序列打出高分（即假阳性）。
2. **OOD (Out-of-Distribution) 性能差**：如果面对的是全新未见过的 pMHC，ERGO 的预测能力往往会断崖式下降。

### 🚀 改进方案
1. **引入现代 Protein Language Models (pLM)**：淘汰 ERGO，使用基于 ESM-2 或 ProtBERT 微调的亲和力预测模型作为基座，这些大模型具备更强的底层生化语义理解能力，鲁棒性更高。
2. **Ensemble 打分机制 (模型集成)**：使用 3-5 个结构不同、初始化随机种子不同的亲和力预测模型进行集成打分。只有当所有模型都认为该 TCR 亲和力高时，才给予高 Reward。这能极大缓解单一模型被 Hack 的问题。
3. **引入基于结构的 Reward (Structural Penalty)**：在强化学习循环外，对高分样本定期进行 AlphaFold/ESMfold 折叠预测，检查生成的 CDR3 环是否确实在空间上能够触及 pMHC 界面，而不是仅仅在 1D 序列层面得分高。

---

## 3. 多目标优化：引入负样本库 (Negative Sampling & Decoy Library)

### 🛑 当前缺陷审计
目前的 `Reward` 计算完全是**单向优化**：只看它对 Target pMHC 的结合力高不高。但在临床应用中，TCR 疗法最致命的风险是**脱靶毒性 (Off-target Toxicity / Cross-reactivity)**。如果生成的 TCR 与自体多肽（Self-peptides）结合，会引发极其严重的自身免疫反应。

### 🚀 改进方案：Contrastive Reward (对比奖励设计)
基于你的思路，我们必须重构 Reward 逻辑，将其设计为多目标优化 (Multi-objective Optimization)：

定义：
- $R_{target}$ = TCR 对目标 pMHC 的亲和力预测得分。
- $\mathcal{D}_{neg}$ = 负样本/诱饵库 (Decoy Library)。包含与 target pMHC 序列相似度极高但绝不能结合的危险肽段，以及常见的人类自身多肽。

**新的 Reward 函数设计**：
$$ R_{total} = \underbrace{R_{target}}_{\text{On-target affinity}} - \alpha \cdot \underbrace{\max_{p_{neg} \in \mathcal{D}_{neg}} R_{off\_target}(TCR, p_{neg})}_{\text{Worst-case cross-reactivity}} - \beta \cdot \underbrace{P_{unnatural}}_{\text{Unnaturalness penalty}} $$

*说明：* 我们使用 $\max$ 函数是因为：只要 TCR 与负样本库中的**任何一个**危险多肽产生了强结合，这个 TCR 废掉。RL Agent 必须学会避开所有“雷区”。

---

## 4. TCR序列的天然性约束 (Naturalness & Developability)

### 🛑 当前缺陷审计
目前使用 AE 重构误差 + GMM 似然度来约束天然性：
```python
# reward.py
seqs, seq_edit_dists, z = self.ae_model.edit_dist(tcrs)
likelihoods = self.gmm_model.score_samples(z)
```
这种做法的局限性在于：AE 和 GMM 的容量太小，且它们是在特定数据集 (TCRdb) 上预训练的。对于稍微新颖一点的突变，AE 可能无法重构，导致 GMM 给出极低的负分，**这会扼杀 RL 探索全新且有效的高亲和力序列的能力**（容易陷入保守的局部最优）。

### 🚀 改进方案：基于 pLM 困惑度的可开发性奖励
使用**蛋白质大语言模型 (如 ESM-2) 的 Perplexity (困惑度/伪似然)** 作为天然性打分，取代老旧的 AE+GMM：
1. **Zero-shot Likelihood**：将生成的 TCR 输入 ESM-2，计算其序列的 Log-Likelihood。ESM-2 在数亿条真实蛋白质上训练过，如果它认为这条序列似然度极低，说明它不符合地球生物的氨基酸演化规律（可能无法折叠、容易聚集或无表达）。
2. **多维理化属性惩罚**：将可开发性 (Developability) 作为硬约束写进 Reward：
   - 惩罚连续的疏水氨基酸 (避免聚集/Aggregation)。
   - 惩罚过度极端的等电点 (pI) 或电荷分布。
   - 惩罚 N-糖基化位点 (N-X-S/T) 的意外生成。

---

## 5. 训练策略与强化学习算法的深层缺陷 (Training Strategies & RL Critics)

除了模型架构和 Reward 设计，TCRPPO 在**如何训练**这个核心环节上也存在明显的不合理之处。这些问题直接导致了模型收敛慢、样本效率低，以及生成的序列“多样性假象”。

### 5.1 MDP 构建与信用分配 (Credit Assignment) 问题
*   **🛑 缺陷审计**：在当前的 `tcr_env.py` 中，环境被建模为一个最多 8 步的定长修改序列 (Edit MDP)。Agent 每次替换一个氨基酸，但在前 7 步中，它**仅仅只收到天然性惩罚**（`reward = min(dist + likelihood - stop, 0)`），只有在最后第 8 步或终止时才一次性拿到 ERGO 亲和力大奖（Terminal Reward）。这导致了严重的**延迟奖励 (Delayed Reward)** 问题，Agent 很难搞清楚究竟是这 8 步中的哪一步（哪一次突变）导致了最终的高亲和力。
*   **🚀 改进方案：Dense Reward Shaping (密集奖励塑造)**
    改用**差分奖励 (Step-wise Delta Reward)**：
    $$ R_{t} = \text{Score}(State_{t}) - \text{Score}(State_{t-1}) $$
    如果当前突变让亲和力提升了，立刻给正反馈；如果亲和力下降，立刻给负反馈。这样能极大提升 PPO 的样本效率和策略网络的收敛速度。

### 5.2 初始状态采样与“盲目探索” (Blind Exploration)
*   **🛑 缺陷审计**：目前的初始状态是从包含数十万条人类 TCR 的数据库 (`TCRdb`) 中随机抽取的 (`self.init_tcr()`)。对于任意一个特定的目标 pMHC，99.9% 的随机 TCR 的初始亲和力几乎为 0。这就好比把你扔到撒哈拉沙漠，让你蒙着眼睛找金矿。Agent 在训练早期会浪费千万次无意义的随机探索（Random Walk）。
*   **🚀 改进方案：目标导向初始化与课程学习 (Guided Initialization & Curriculum Learning)**
    1. **Homologous Initialization (同源初始化)**：不要完全随机抽。对于目标靶点，如果在 VDJdb 中已有几条已知的弱结合 TCR，将它们作为种子序列进行突变。
    2. **Active Curriculum (主动课程学习)**：起初，放宽亲和力阈值（让 Agent 尝到一点甜头），然后随着训练轮数 (Epochs) 的增加，逐渐收紧 `score_stop_criteria`，逼迫 Agent 攀登更高的亲和力山峰。

### 5.3 状态表征的贫乏 (State Representation)
*   **🛑 缺陷审计**：当前 Observation Space 使用简单的数字索引 (1~20 代表氨基酸) 输入给浅层的 LSTM (`SeqEmbed`) 提取特征。这种 1D 的字符级表征完全抛弃了氨基酸的生化相似性，更无法捕捉 CDR3 环与 MHC-Peptide 结合界面的 3D 空间交互。
*   **🚀 改进方案：冻结的 pLM 嵌入作为状态 (Frozen pLM Embeddings)**
    将环境状态 (State) 升级为特征向量：当 TCR 序列发生突变时，先将其通过冻结的轻量级蛋白质语言模型 (如 ESM-2 35M)，取最后一层的 CLS token 向量或 mean-pooling 向量作为 RL 策略网络 (Actor-Critic) 的输入。这等于给 PPO 装上了一副“懂生物化学”的眼睛，网络只需学习如何映射高维生化特征到动作概率。

### 5.4 算法选型：PPO vs GFlowNets
*   **🛑 缺陷审计**：PPO 是为了寻找**单一最优策略** (Single Optimal Policy) 而设计的。在药物设计/TCR设计中，PPO 的致命伤是**模式坍缩 (Mode Collapse)**——一旦它发现了一个能拿到高分（比如 ERGO 0.95）的序列模式（比如 `ASSY....F`），它会为了最大化期望收益，不断地输出这个高度相似的序列族，导致生成的序列极其单一。这对需要建立大规模“抗体/TCR候选库”的后续湿实验来说是灾难。
*   **🚀 改进方案：生成流网络 (GFlowNets)**
    药物生成领域目前的 SOTA 算法正从 RL(PPO) 转向 **GFlowNets (Generative Flow Networks)**。GFlowNets 并不寻求最大化奖励，而是**使得生成某条序列的概率与其 Reward 成正比 ($P(x) \propto R(x)$)**。
    这意味着：如果世界上有 100 种截然不同但亲和力都很高的 TCR，GFlowNet 会等概率地把它们全部采样出来，而 PPO 只会死死咬住其中最好或者最先发现的那 1 种。引入 GFlowNet 思想或在 PPO 中加入强烈的**新颖性搜索 (Novelty Search / Intrinsic Motivation)** 是保证候选库多样性的终极解法。

---

## 6. Next Steps: 完整工程实施路径 (Roadmap)

如果你决定开始重构，我们的第一阶段目标是搭建一个**“插件化” (Pluggable) 的 RL 框架**，让你能够随时插拔不同的 Reward 模块和数据集。

* **Step 1: 环境层重构 (Environment Refactoring)**
  - 修改 `code/tcr_env.py`，实现一个基于 `<PAD>` token 的动态 Indel 动作空间。
* **Step 2: Reward API 抽象化 (Reward Decoupling)**
  - 重写 `reward.py`，设计一个标准的 `RewardManager` 接口。
  - 实现三个组件：`TargetAffinityScorer`, `DecoyToxicityScorer`, `NaturalnessScorer`。
* **Step 3: 引入你自己的模型 (Integration)**
  - 将你准备好的新模型封装到上述 Scorer 中。准备好负样本库集（CSV或FASTA格式）。
* **Step 4: PPO 训练验证 (Sanity Check)**
  - 用一个小批量的靶点和极其严苛的负样本惩罚进行过拟合测试，观察 PPO 能否成功在避开负样本的同时找到高分突变。

