# tFold Peptide Input Analysis

## 问题背景

tcrppo_v2 使用 ESM-2 embeddings 作为 peptide input，但发现 20 个训练 peptides 的 embeddings 高度相似（mean cosine similarity = 0.9685），导致网络难以区分不同 peptides。

现在分析 tFold 如何处理 peptide input，寻找改进方案。

---

## tFold 的 Peptide Input 处理方式

### 1. **不使用单独的 Peptide Embeddings**

tFold **不使用** ESM-2 对 peptide 单独编码的 embeddings。相反，它使用：

#### (a) Structure-based Features (sfea)
- **来源**: tFold-TCR 结构预测模型的中间层特征
- **维度**: `[L, 192]` 其中 L 是序列总长度（TCRβ + TCRα + peptide + MHC）
- **提取方式**: 
  ```python
  sfea = data["raw_sfea"].float()  # [L, 192]
  sfea_pep = sfea[pep_range[0]:pep_range[1]]  # 只取 peptide 部分
  ```
- **关键特性**: 
  - sfea 是 **structure-aware** 的，包含了 peptide 在 pMHC 复合物中的结构信息
  - 不是纯序列 embedding，而是考虑了 3D 结构和相互作用的特征

#### (b) Pairwise Features (pfea)
- **来源**: tFold-TCR 的 pair representation
- **维度**: `[L, L, 128]` 全局 pair features
- **提取方式**: 只保留 interface 相关的子块
  ```python
  pfea_cdr3b_pep = pfea[cdr3b_start:cdr3b_end, pep_start:pep_end, :]  # [Lb, Lp, 128]
  pfea_cdr3a_pep = pfea[cdr3a_start:cdr3a_end, pep_start:pep_end, :]  # [La, Lp, 128]
  ```
- **关键特性**:
  - 编码了 **CDR3-peptide 残基对** 之间的相互作用信息
  - 每个 (CDR3_residue, peptide_residue) pair 有 128-dim 特征
  - 包含距离、角度、接触概率等结构信息

#### (c) Cα Coordinates (mono_ca)
- **来源**: tFold-TCR 预测的 Cα 原子坐标
- **维度**: `[L, 3]` (x, y, z 坐标)
- **提取方式**:
  ```python
  mono_ca_pep = mono_rec[pep_range[0]:pep_range[1], 1, :]  # [Lp, 3]
  ```
- **用途**: 计算 CDR3-peptide 残基间的欧氏距离，用 RBF encoding 作为 attention bias

---

### 2. **Structure-Aware RPIM (Residue-Pair Interaction Module)**

tFold 的 classifier 使用 **RPIM** 来融合 TCR 和 peptide 信息：

```python
class StructureAwareRPIM(nn.Module):
    def __init__(self, d_sfea=192, d_int=128, n_heads=4, pfea_dim=128, n_rbf=16):
        # 1. Project sfea to interaction space
        self.cdr3_proj = nn.Sequential(nn.Linear(d_sfea, d_int), nn.GELU())
        self.pep_proj = nn.Sequential(nn.Linear(d_sfea, d_int), nn.GELU())
        
        # 2. Multi-head bilinear interaction
        self.bilinear_weight = nn.Parameter(torch.randn(n_heads, d_head, d_head))
        
        # 3. Structure biases
        self.rbf_enc = RBFDistanceEncoding(n_rbf=16, cutoff=50.0)
        self.dist_bias = nn.Linear(n_rbf, n_heads)  # distance → attention bias
        self.pfea_bias = nn.Sequential(
            nn.Linear(pfea_dim, pfea_dim // 2),
            nn.ReLU(),
            nn.Linear(pfea_dim // 2, n_heads)
        )  # pfea → attention bias
        
        # 4. Attention pooling
        self.row_attn = nn.Sequential(...)  # CDR3 residue importance
        self.col_attn = nn.Sequential(...)  # peptide residue importance
```

**Forward pass**:
```python
def forward(self, cdr3_sfea, pep_sfea, cdr3_mask, pep_mask, ca_cdr3, ca_pep, pfea):
    # 1. Project to interaction space
    cdr3 = self.cdr3_proj(cdr3_sfea)  # [B, Nc, d_int]
    pep = self.pep_proj(pep_sfea)      # [B, Np, d_int]
    
    # 2. Multi-head bilinear interaction
    interaction = bilinear(cdr3, pep)  # [B, n_heads, Nc, Np]
    
    # 3. Add structure biases
    dist = torch.cdist(ca_cdr3, ca_pep)  # [B, Nc, Np]
    rbf = self.rbf_enc(dist)             # [B, Nc, Np, n_rbf]
    d_bias = self.dist_bias(rbf)         # [B, Nc, Np, n_heads]
    interaction += d_bias.permute(0, 3, 1, 2)
    
    p_bias = self.pfea_bias(pfea)        # [B, Nc, Np, n_heads]
    interaction += p_bias.permute(0, 3, 1, 2)
    
    # 4. Bidirectional attention pooling
    row_agg = softmax_pool(interaction, dim=-1)  # CDR3 ← peptide
    col_agg = softmax_pool(interaction, dim=-2)  # peptide ← CDR3
    
    # 5. Weighted pooling by residue importance
    row_pooled = attention_pool(row_agg)  # [B, d_int]
    col_pooled = attention_pool(col_agg)  # [B, d_int]
    
    return cat([row_pooled, col_pooled])  # [B, 2*d_int]
```

**关键设计**:
- **不是简单拼接 TCR + peptide embeddings**
- 而是计算 **residue-level interaction matrix**，每个 (CDR3_res, pep_res) pair 有独立的 attention score
- Structure biases (distance, pfea) 直接注入到 attention 中，强制网络关注结构上接近的残基对
- 双向 pooling: CDR3 看 peptide，peptide 看 CDR3

---

### 3. **为什么 tFold 的方法有效？**

#### (a) Peptide 信息是 **context-dependent** 的
- sfea 不是孤立的 peptide embedding，而是 peptide **在 pMHC 复合物中** 的特征
- 同一个 peptide 在不同 MHC 上的 sfea 会不同（因为结构不同）
- 这避免了 ESM-2 的问题：ESM-2 只看序列，不看结构和相互作用

#### (b) Residue-level interaction 提供细粒度信息
- 不是 pool 成一个 peptide vector，而是保留每个残基的信息
- CDR3 的不同位置可以关注 peptide 的不同位置
- 例如：CDR3β 可能主要看 peptide 的 N-端，CDR3α 看 C-端

#### (c) Structure biases 强制网络学习物理合理的相互作用
- Distance bias: 空间上远的残基对 attention 低
- Pfea bias: tFold 预测的相互作用强度直接影响 attention
- 这些 inductive biases 减少了需要从数据中学习的自由度

---

## tcrppo_v2 的问题诊断

### 当前架构 (policy.py)
```python
# Observation: [TCR_emb(1280) | pMHC_emb(1280) | scalars(2)] = 2562-dim
obs = torch.cat([tcr_emb, pmhc_emb, scalars], dim=-1)
backbone = Linear(2562 → 512) → ReLU → Linear(512 → 512) → ReLU
```

**问题**:
1. **pMHC_emb 是 ESM-2 对整个 pMHC 序列的 mean pooling**
   - 丢失了 peptide 的 residue-level 信息
   - 不同 peptides 的 ESM-2 embeddings 高度相似（cos-sim 0.9685）
   
2. **Episode 内 peptide 是常量**
   - 网络学会 shortcut: 只看 TCR 的变化，忽略 peptide
   - Peptide pathway 严重衰减（sensitivity 0.08-0.16）

3. **没有结构信息**
   - ESM-2 是纯序列模型，不知道 TCR-peptide 如何相互作用
   - 网络需要从头学习 binding physics

### Cross-Attention 方案 (policy_cross_attn.py)
```python
tcr_enc = Linear(1280 → 512)
pep_enc = Linear(1280 → 512)
fusion = CrossAttentionFusion(tcr_enc, pep_enc, n_heads=4)
```

**改进**:
- 强制 TCR 和 peptide 通过 attention 交互
- 但仍然使用 **pooled ESM-2 embeddings**，没有 residue-level 信息

**局限**:
- 如果 20 个 peptides 的 embeddings 都很相似，cross-attention 也无法区分它们
- 没有结构信息，attention 是 data-driven 的，需要大量数据

---

## 改进方案建议

### 方案 1: **Residue-Level Peptide Representation** (推荐)

不使用 pooled pMHC embedding，而是保留 peptide 的 per-residue embeddings：

```python
# Input
tcr_seq = "CASSLGQAYEQYF"  # CDR3β
pep_seq = "GILGFVFTL"       # peptide

# ESM-2 encoding
tcr_emb = esm_model(tcr_seq)  # [L_tcr, 1280]
pep_emb = esm_model(pep_seq)  # [L_pep, 1280]

# Policy architecture
class ActorCriticResidueLevel(nn.Module):
    def __init__(self):
        self.tcr_proj = nn.Linear(1280, 256)
        self.pep_proj = nn.Linear(1280, 256)
        self.interaction = nn.MultiheadAttention(256, num_heads=4)
        self.tcr_pool = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)  # attention weights
        )
        self.pep_pool = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
    def forward(self, obs):
        tcr_emb = obs["tcr_residues"]  # [B, L_tcr, 1280]
        pep_emb = obs["pep_residues"]  # [B, L_pep, 1280]
        
        tcr_feat = self.tcr_proj(tcr_emb)  # [B, L_tcr, 256]
        pep_feat = self.pep_proj(pep_emb)  # [B, L_pep, 256]
        
        # TCR attends to peptide
        tcr_ctx, _ = self.interaction(
            query=tcr_feat.transpose(0, 1),
            key=pep_feat.transpose(0, 1),
            value=pep_feat.transpose(0, 1)
        )  # [L_tcr, B, 256]
        tcr_ctx = tcr_ctx.transpose(0, 1)  # [B, L_tcr, 256]
        
        # Attention pooling
        tcr_weights = F.softmax(self.tcr_pool(tcr_ctx).squeeze(-1), dim=1)  # [B, L_tcr]
        tcr_pooled = (tcr_ctx * tcr_weights.unsqueeze(-1)).sum(dim=1)  # [B, 256]
        
        pep_weights = F.softmax(self.pep_pool(pep_feat).squeeze(-1), dim=1)  # [B, L_pep]
        pep_pooled = (pep_feat * pep_weights.unsqueeze(-1)).sum(dim=1)  # [B, 256]
        
        fused = torch.cat([tcr_pooled, pep_pooled, scalars], dim=-1)
        # ... rest of policy heads
```

**优势**:
- 保留 peptide 的 residue-level 信息，不同位置的氨基酸可以独立影响 TCR
- Attention 可以学习哪些 peptide 位置对 binding 最重要
- 即使 pooled embeddings 相似，residue-level 的差异仍然可以被利用

**挑战**:
- Observation space 变大：需要存储 per-residue embeddings
- 需要处理变长序列（padding + masking）

---

### 方案 2: **Augmented Peptide Features**

在 ESM-2 embeddings 基础上添加额外特征：

```python
def augment_peptide_features(pep_seq, pep_emb):
    """
    Args:
        pep_seq: str, e.g., "GILGFVFTL"
        pep_emb: [1280] ESM-2 mean-pooled embedding
    
    Returns:
        [1280 + N_aug] augmented features
    """
    # 1. Amino acid composition (20-dim one-hot sum)
    aa_comp = compute_aa_composition(pep_seq)  # [20]
    
    # 2. Physicochemical properties
    hydrophobicity = compute_hydrophobicity(pep_seq)  # scalar
    charge = compute_net_charge(pep_seq)              # scalar
    aromaticity = compute_aromaticity(pep_seq)        # scalar
    
    # 3. Position-specific features
    n_term_aa = one_hot(pep_seq[0])   # [20] N-terminal residue
    c_term_aa = one_hot(pep_seq[-1])  # [20] C-terminal residue
    anchor_p2 = one_hot(pep_seq[1]) if len(pep_seq) > 1 else zeros(20)  # P2 anchor
    anchor_p9 = one_hot(pep_seq[-1])  # P9 anchor (for 9-mers)
    
    # 4. Length encoding
    length_feat = [len(pep_seq), len(pep_seq)**2]  # [2]
    
    # Concatenate
    aug_feat = torch.cat([
        pep_emb,           # [1280]
        aa_comp,           # [20]
        torch.tensor([hydrophobicity, charge, aromaticity]),  # [3]
        n_term_aa,         # [20]
        c_term_aa,         # [20]
        anchor_p2,         # [20]
        torch.tensor(length_feat)  # [2]
    ])  # [1280 + 85] = 1365-dim
    
    return aug_feat
```

**优势**:
- 简单，不改变 observation space 结构
- 添加 ESM-2 缺失的显式特征（anchor positions, physicochemical properties）
- 这些特征对 MHC binding 很重要，可能帮助网络区分 peptides

**局限**:
- 仍然是 pooled representation，丢失了 residue-level 信息
- 如果 ESM-2 embeddings 已经包含这些信息（只是不显式），augmentation 可能帮助不大

---

### 方案 3: **Multi-Peptide Episodes** (训练策略改进)

不改变 model，而是改变训练数据分布：

```python
class MultiPeptideEnv:
    def __init__(self, peptide_pool):
        self.peptide_pool = peptide_pool  # 20 peptides
        
    def reset(self):
        # 每个 episode 随机选择一个 peptide
        self.current_peptide = random.choice(self.peptide_pool)
        self.current_pmhc_emb = self.get_pmhc_embedding(self.current_peptide)
        # ... initialize TCR
        
    def step(self, action):
        # TCR 变化，但 peptide 保持不变（episode 内）
        # ... apply action to TCR
        obs = self.get_observation()  # includes self.current_pmhc_emb
        reward = self.compute_reward()
        return obs, reward, done, info
```

**训练时**:
- 每个 episode 使用不同的 peptide
- 网络被迫学习 peptide-dependent policy
- 因为不同 episodes 的 optimal TCR 不同（取决于 peptide）

**优势**:
- 不改变 model 架构
- 强制网络使用 peptide 信息（否则无法泛化到不同 peptides）

**局限**:
- 如果 peptide embeddings 太相似，网络仍然可能学到 peptide-agnostic policy
- 需要更多 episodes 才能收敛（因为每个 episode 的 peptide 不同）

---

### 方案 4: **Auxiliary Peptide Prediction Loss**

添加辅助任务强制网络学习 peptide representation：

```python
class ActorCriticWithPeptideHead(nn.Module):
    def __init__(self, n_peptides=20):
        # ... original policy heads
        self.peptide_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, n_peptides)  # classify which peptide
        )
        
    def forward(self, obs):
        # ... backbone
        hidden = self.backbone(obs)  # [B, 512]
        
        # Policy heads
        op_logits = self.op_head(hidden)
        pos_logits = self.pos_head(hidden)
        tok_logits = self.tok_head(hidden)
        value = self.value_head(hidden)
        
        # Auxiliary peptide prediction
        pep_logits = self.peptide_classifier(hidden)  # [B, 20]
        
        return op_logits, pos_logits, tok_logits, value, pep_logits

# Training loss
ppo_loss = compute_ppo_loss(...)
peptide_loss = F.cross_entropy(pep_logits, peptide_labels)
total_loss = ppo_loss + 0.1 * peptide_loss  # auxiliary loss weight
```

**优势**:
- 强制 backbone 学习区分不同 peptides 的 representation
- 即使 peptide embeddings 相似，网络也必须找到区分它们的方式
- Auxiliary loss 提供额外的 supervision signal

**局限**:
- 需要 peptide labels（但这在 tcrppo_v2 中是已知的）
- 如果 peptide embeddings 真的无法区分，auxiliary loss 也无法强制网络学到有用的特征

---

## 推荐方案

### 短期（立即可行）: **方案 2 + 方案 4**
1. **Augment peptide features** with anchor positions, physicochemical properties
2. **Add auxiliary peptide classification loss** to force network to use peptide info

**实现步骤**:
1. 修改 `get_pmhc_embedding()` 添加 augmented features
2. 修改 `ActorCritic` 添加 `peptide_classifier` head
3. 修改 `ppo_trainer.py` 添加 auxiliary loss
4. 重新训练，监控 peptide classification accuracy

### 中期（需要重构）: **方案 1**
1. **Residue-level peptide representation** with cross-attention
2. 修改 observation space 存储 per-residue embeddings
3. 实现 residue-level attention pooling

**实现步骤**:
1. 修改 `esm_cache.db` 存储 per-residue embeddings（不只是 mean pooling）
2. 实现 `ActorCriticResidueLevel` policy
3. 修改 `TCRDesignEnv` 返回 residue-level observations
4. 重新训练

### 长期（理想方案）: **集成 tFold features**
1. 使用 tFold-TCR 预测 TCR-pMHC 结构
2. 提取 sfea, pfea, Cα coordinates
3. 实现 Structure-Aware RPIM 作为 policy backbone

**挑战**:
- 需要在每个 RL step 调用 tFold（计算开销大）
- 或者预计算所有可能 TCR-pMHC 的 features（存储开销大）
- 可能需要 tFold 的 differentiable 版本（用于 policy gradient）

---

## 总结

**tFold 的核心洞察**:
1. **Peptide 不应该被 pool 成单个 vector**，residue-level 信息很重要
2. **Structure matters**: 序列相似的 peptides 可能有不同的结构和相互作用模式
3. **Interaction is key**: TCR-peptide binding 是 residue-pair 的相互作用，不是两个 vectors 的拼接

**tcrppo_v2 的改进方向**:
1. **立即**: Augment features + auxiliary loss（低成本，可能有帮助）
2. **中期**: Residue-level representation（需要重构，但更符合 binding physics）
3. **长期**: 集成 structure features（理想方案，但工程挑战大）

**下一步行动**:
- 先实现方案 2 + 4，快速验证是否有改进
- 如果有效，再投入资源实现方案 1
- 如果方案 1 仍不够，考虑集成 tFold features（但这可能超出 RL 项目的范围）
