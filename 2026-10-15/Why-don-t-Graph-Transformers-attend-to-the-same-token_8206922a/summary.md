---
title: "Why-don-t-Graph-Transformers-attend-to-the-same-token"
source: https://arxiv.org/pdf/2610.19324v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-02 02:13:39"
---

# 论文速读：Why-don-t-Graph-Transformers-attend-to-the-same-token

## 一句话总结
本文系统证明图 Transformer 极少在分子图上形成类似 LLM 的 attention sink，其根本原因并非图结构或深度，而是注意力打分函数：将 GRIT 的门控打分替换为点积后，sink 率从 0.090 跃升至 0.680；点积模型的汇聚节点对预测近乎无用，而 GRIT 的高关注度节点具有明确的预测功能性。

## 研究问题与动机
1. **LLM 中的 attention sink 机制已明确**：大型语言模型中注意力强烈汇聚至首 token（如 `<BOS>`），被认为可防止深层模型因 softmax 扩散而过度混合（over-mixing）。
2. **图 Transformer 面临相似压力但行为反常**：图深度加深会引发过平滑（over-smoothing），全局注意力进一步加剧该问题；理论上可能需要一个“汇点”缓冲，但普通图无起始 token，且置换等变性禁止模型偏心不可区分的节点。
3. **现有工作缺乏图域归因**：已有研究仅指出图缺乏自然汇点，但未解释 sink 究竟能在图的何处形成，也未厘清是图结构约束还是架构组件决定了 sink 是否出现。
4. **动机**：以 SOTA 图 Transformer GRIT 为基准，定量剥离图对称性/局部性与打分函数对 sink 的贡献，回答“图 Transformer 为何不汇聚于同一 token”这一机制性问题。

## 核心贡献（创新点）
1. **首次系统量化图 Transformer 的 attention sink 现象并发现反 LLM 趋势**。与仅描述 LLM 汇点的已有工作不同，本文在图域建立可比度量
