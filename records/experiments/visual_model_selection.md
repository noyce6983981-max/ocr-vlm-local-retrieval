# 视觉检索模型选型

日期：2026-07-29

## 决策

下一阶段优先实测官方`Qwen/Qwen3-VL-Embedding-2B`，不先使用通用聊天模型生成图片描述。

## 选择理由

- 2026年发布，能体现对视觉语言模型和多模态检索热点的关注。
- 官方定位就是图片、截图、视觉文档与文本之间的检索。
- 使用双塔结构，可提前编码图片，查询时只需向量相似度计算。
- 2B参数版本比8B版本更适合RTX 4060 8GB显存。
- 支持中文和ModelScope下载，适合本项目运行环境。
- 最多输出2048维向量，可直接接入FAISS并与现有BGE-M3基线比较。

官方资料：

- GitHub：<https://github.com/QwenLM/Qwen3-VL-Embedding>
- ModelScope：<https://modelscope.cn/models/qwen/Qwen3-VL-Embedding-2B>

## 为什么不直接使用通用VLM聊天Demo

本项目的核心任务是检索。专用多模态Embedding模型能直接输出可评价的向量和Top-K结果，更容易设计Recall@K、MRR、延迟和显存对照实验。通用VLM后续可用于结果解释或问答，但不作为第一版检索核心。

## 计划对照

| 组别 | OCR文本分支 | 视觉分支 | 融合方式 |
|---|---|---|---|
| A | BGE-M3 | 无 | 文本基线 |
| B | 无 | Qwen3-VL-Embedding-2B | 视觉基线 |
| C | BGE-M3 | Qwen3-VL-Embedding-2B | 固定权重 |
| D | BGE-M3 | Qwen3-VL-Embedding-2B | OCR质量自适应权重 |

评价指标：Recall@1、Recall@3、MRR、单图编码时间、查询时间、峰值显存。

## 8GB显存风险控制

- 新建独立`.venv-vl`，不破坏现有OCR和BGE-M3环境。
- FP16或BF16、batch size 1、限制输入图片像素。
- OCR、BGE-M3和视觉模型分阶段运行，用文件传递向量。
- 是否能稳定运行、实际峰值显存和速度均以实测为准，不提前宣称。
