# 自动查询理解与混合检索 V12 验收记录

日期：2026-08-01

## 改造目标

- 删除独立“探索发现”搜索模式，避免用户先理解技术概念再选模式。
- 快速与深度搜索都自动判断证据需求，并适量融合发现能力。
- 不再用“字数 + 姓氏首字”单独判人名，避免“山水、林业、方程式”等误判。
- 保留“没有证据就不返回”的底线，但不再用同一个高阈值拦截所有查询。

## 查询理解

系统将输入划分为五类证据需求：

1. `entity_exact`：人名/短实体，只接受 OCR 或元数据精确命中。
2. `text_evidence`：字段、编号、课程、地址等事实查询，执行校准后的拒答门控。
3. `visual_discovery`：山水、森林、沙漠等视觉主题，采用低门槛召回优先策略。
4. `topic_discovery`：技术和一般主题，使用 OCR 语义 + BM25，并设置弱相关下限。
5. `visual_metadata` / `mixed`：显式图片形式或复杂问句，按需组合多路证据。

分类信号包括明确查询词、视觉形式词、技术语境、中文词性、人物上下文和
保守姓氏回退；不再依赖单一规则。版本化回归集位于
`data/evaluation/search_intent_regression_v12.csv`。

## 两档产品行为

- 快速搜索：自动路由后执行所需分支；视觉主题使用
  `0.78 × Image + 0.15 × Dense + 0.07 × BM25`。
- 多模态完整核验：复用同一候选召回，再用 Qwen3-VL Reranker 对候选页面重排。
- 独立探索模式：已从网页和 `live_search.py` 命令行选项中删除。
- 低置信度：普通弱相关结果默认隐藏，可人工展开排查；人名无精确证据时彻底隐藏。

## 成熟检索机制参考

- Algolia Detecting Intent：从查询中识别意图并动态调整参数、过滤条件与查询。
  https://www.algolia.com/doc/guides/managing-results/rules/detecting-intent
- Algolia Query Categorization：区分 broad、narrow、ambiguous、none，并结合置信度。
  https://www.algolia.com/doc/guides/algolia-ai/query-categorization
- Azure Hybrid Search：全文和向量检索并行，再使用 RRF 合并；姓名、编号等仍适合精确关键词证据。
  https://learn.microsoft.com/en-us/azure/search/hybrid-search-overview
- Azure Semantic Ranker：先做一级召回，再对候选执行二级语义重排。
  https://learn.microsoft.com/en-us/azure/search/semantic-search-overview

## 真实查询校准

| 查询 | 自动理解 | 默认结果 | 关键信号 | 本次耗时 |
| --- | --- | --- | --- | --- |
| 山水 | 视觉主题发现 | 接受，前五为山景/山林/河景 | 视觉 0.578 / 门槛 0.350 | 26.6 秒 |
| 沙漠 | 视觉主题发现 | 接受，前三为荒漠植被或沙丘 | 视觉 0.476 / 门槛 0.350 | 24.9 秒 |
| 量子泡沫弦理论 | 技术主题 | 拒绝，弱候选默认隐藏 | 语义 0.429 / 门槛 0.440 | 15.2 秒 |
| 盛和 | 人名/短实体 | 拒绝，不运行模型且无候选 | 精确命中 0 / 门槛 1 | 0.23 秒 |

`山水`多模态完整核验的前五仍均为相关山景/河景；OCR 语义、BM25、视觉语义和
Qwen3-VL 重排分支均执行，最近一次完整运行耗时 15.25 秒。

## 验证状态

- 查询路由与检索策略定向测试：40 项通过。
- 全量自动化测试：156 项通过。
- 主检索站和独立纠错站健康检查均为 HTTP 200。
- 浏览器验收：模式列表仅保留 OCR、图片、快速、完整核验四项；不存在独立探索模式。
- 浏览器实测“山水”：快速模式显示三路融合，Top 3 均为相关自然图像；完整核验显示四路参与与多模态重排结果。
- 浏览器实测“盛和”：无模型分支、无候选；“量子泡沫弦理论”：默认隐藏弱候选并提供人工排查开关。
