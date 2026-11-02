---
title: "When-Does-Momentum-Help-in-Reinforcement-Learning-with-Verif"
source: https://arxiv.org/pdf/2610.20413v1.pdf
model: agnes-2.5-flash
chunks: 3
summarized_at: "2026-11-02 02:12:08"
---

# 论文速读：When-Does-Momentum-Help-in-Reinforcement-Learning-with-Verif

## 一句话总结
本文从理论与实验双重视角揭示了EMA动量在RLVR及大模型post-training中的失效机制与修复方案：证明固定动量系数β在小学习率下会被采样噪声淹没，提出将β按η/(1−β)=Θ(1)动态缩放的准则，并在Bandit模拟与Qwen系列模型上验证该策略可恢复动量的加速效果并显著降低训练方差。

## 研究问题与动机
1. RLVR/post-training中动量优化器（如AdamW的一阶动量）实际表现不稳定，有时SGD反而能匹配甚至超越带动量的变体（Mukherjee et al., 2026）。
2. 现有工作缺乏对“何时动量有效”的严谨理论解释，实践中多依赖经验调参（如固定β=0.9）。
3. 小学习率是RLVR稳定训练的关键，但固定β下学习率降低会导致动量期望增益衰减快于采样
