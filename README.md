- 数据路径： `/fast/data/scTFM`
- 模型路径：`/fast/projects/scTFM`

## 自编码器（setAE）
```plaintext
  SetSCAE (统一接口)
      ├── GraphSetAE      (GAT + 链路预测)
      ├── ContrastiveSetAE (对比学习)
      └── MaskedSetAE     (掩码预测)
```
### 方案
|              | GraphSetAE              | ContrastiveSetAE    | MaskedSetAE                                  |
|--------------|-------------------------|---------------------|----------------------------------------------|
| 编码器          | input_proj + GAT layers | cell_encoder (scVI) | input_proj + Transformer                     |
| 编码方式         | 图注意力聚合邻居信息              | 独立编码每个细胞            | 掩码中心 + 上下文编码                                 |
| 特有模块         | gat_layers              | projection_head     | mask_token, context_encoder, prediction_head |
| cell_encoder | 冻结 (不使用)                | 使用                  | 冻结 (不使用)                                     |
| 特有损失         | 链路预测 (adj_pred)         | InfoNCE 对比损失        | 掩码基因预测 (MSE)                                 |
### Loss
所有策略共享：NB 重建损失 (所有细胞)
recon_loss = -log_nb_positive(target, mu_all, theta_all).mean()
策略特定损失：

| 策略          | 特定损失    | 公式                             |
|-------------|---------|--------------------------------|
| Graph       | 链路预测    | BCE(adj_pred, adj)             |
| Contrastive | InfoNCE | CrossEntropy(pos_sim, neg_sim) |
| Masked      | 掩码预测    | MSE(pred_genes[mask], x[mask]) |

### 2026-01-14

- /fast/projects/scTFM/models/ae/2026-01-16_04-13-53
  - z_all 包含所有细胞的 latent (batch, set_size, n_latent)
  - 但 decode_cell 只解码中心细胞
  - 重建损失只计算中心细胞

- 之前：每个 batch 只训练 batch_size 个细胞（中心细胞）
- 现在：每个 batch 训练 batch_size × set_size 个细胞

应该是对于每一个细胞，都要被选择作为一次中心细胞，然后进行微环境建模学习
- 同一个 tissue 内随机采样就是微环境
- 每个 forward pass 中，轮流让 bag 中的每个细胞作为中心，计算所有细胞的重建损失。
    - 折中方案，如果每个细胞再随机采的话太慢了


## RTF
### 2026-01-13
不再使用`split_label`，训练的时候划分
- `/fast/projects/scTFM/models/rtf/2026-01-15_10-07-57`