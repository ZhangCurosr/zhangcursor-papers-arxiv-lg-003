---
title: "Temporal-Visuo-Tactile-Learning-for-Dexterous-Grasp-Stabilit"
source: https://arxiv.org/pdf/2610.10283v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-10-08 10:12:59"
---

# 论文速读：Temporal-Visuo-Tactile-Learning-for-Dexterous-Grasp-Stabilit

## 一句话总结
本文系统性研究了高分辨率动态触觉对多指灵巧手抓握稳定性预测的贡献，收集了涵盖200个物体、10,000次抓取的 multimodal 数据集，训练了端到端时序多模态稳定性预测器，并将其作为在线 lift-or-regrasp 门控部署于真机，使有效执行抬升的成功率较纯视-本体觉门控提升 10.5 个百分点。

## 研究问题与动机
- 灵巧手稳定抓取本质上是接触动态交互问题，稳定性取决于闭合过程中的接触形成、载荷分配与抬升响应，仅依赖外部视觉位姿选择难以可靠预测。
- 现有视-触抓取研究多聚焦并行夹爪或静态触觉快照，对多手指场景下各传感模态的信息量、触觉空间细节与时间动态如何量化贡献缺乏系统结论。
- Digit 360 等高分辨率多模态指尖传感器日益普及，但其相机流、音频、IMU、压力流在稳定性预测中的实际价值
