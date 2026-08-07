# V16：分路线证据门控与历史盲测回归

日期：2026-08-01

## 结论边界

V16 完成了检索路由、接收门槛、复合查询保护和分路线排序的第一阶段改造。结果表明此前的低相关结果主要不是单一权重问题，而是“排名使用一套证据、接收判断使用另一套证据”。

本记录复用了已经参与过分析的 Blind V1 与 legacy99，因此只能作为历史回归和开发诊断，不能作为全新的独立盲测成绩。Blind V1 中 57/60 条标签由 AI 辅助完成、3/60 条由用户人工完成，也不能据此宣称人工标注质量。

## 根因

1. `text_evidence` 路线跳过视觉分支后，原三路开放集门控不再适配，BM25 的局部词命中可能单独接收错误候选。
2. `topic_discovery` 曾把浏览式请求近似视为天然可回答，弱语义命中也容易放行。
3. `visual_metadata` 排名主要依赖文件名和元数据，但接收门槛未验证同一候选上的视觉、文字与元数据证据是否对齐。
4. 复合查询保护执行过早，后续发现和重排逻辑可能覆盖其拒绝结果。
5. 引号中的多个短语曾按候选并集判断，可能出现短语分别落在不同资料中却被当成完整证据。

## 实现

1. 检索策略升级到 `SEARCH_POLICY_VERSION = 16`，排序配置独立固化在 V16 目录，内容缓存与策略版本继续分离。
2. 文字证据路线要求同一候选同时具备语义与 BM25 证据；完整精确引文可作为更强的替代证据。
3. 多个引文必须在同一资料中全部出现，不能用不同资料的命中并集通过门槛。
4. 视觉元数据路线要求同候选视觉证据；菜单、通知、界面等字面型请求还需文字证据或完整精确引文。
5. “去年、昨晚”等相对时间或个人语境必须在索引证据中出现，避免依靠常识或弱相似度臆答。
6. “旁边、戴着、放着、上写着”等关系型复合请求在所有发现和重排逻辑之后执行最终视觉门槛，不能再被覆盖。
7. 颜色、普通视觉、视觉元数据、混合、主题发现分别选择适合自身证据结构的排序源。
8. 结果解释改为按路线展示，视觉元数据结果明确显示“文件名/元数据 + 视觉内容”，并补充元数据分和主题精确命中数。
9. 新增紧凑特征采集脚本，能在不保存整库排名的情况下比较文字、视觉、BM25、RRF、质量融合与自适应排序。

## Blind V1 历史回归

| 指标 | V15 | V16 | 预锁目标 |
|---|---:|---:|---:|
| Ranker R@1 | 27.50% | 52.50% | ≥ 50% |
| Ranker R@10 | 65.00% | 85.00% | ≥ 80% |
| 可回答接收率 | 100.00% | 100.00% | ≥ 90% |
| 无答案拒绝率 | 10.00% | 70.00% | ≥ 70% |
| 端到端 Top-1 | 21.67% | 58.33% | — |
| 热缓存 P95 | 0.723 秒 | 0.915 秒 | ≤ 1.5 秒 |

60 条查询全部命中组件缓存。V16 仍有 6 条无答案误接收、13 条 Top-1 排名错误和 6 条 Top-10 召回遗漏，说明下一阶段仍应重点提高必要属性覆盖和强负例辨别能力，而不是继续只调全局融合权重。

## legacy99 交叉检查

旧的 99 条正式查询同样已经被开发过程看过，只用于检查兼容性。最终重采集结果另存为 `data/evaluation/v16/development/legacy99_v16_final_features.csv`：

| 分片 | 可回答 R@1 | 可回答 R@10 | 可回答接收 | 无答案拒绝 |
|---|---:|---:|---:|---:|
| train | 49/54 | 54/54 | 54/54 | 5/6 |
| validation | 17/17 | 17/17 | 17/17 | 2/2 |
| test | 18/18 | 18/18 | 18/18 | 2/2 |

其中曾唯一误拒的 `rq100_047` 是包含两个引文的场景图片；V16 修正为“两个引文必须在同一资料中完整命中”后，该查询恢复接收。最终 99 条中只有 `rq100_055` 紫色热气球仍被误接收，明天应加入新的颜色/物体强负例校准，而不针对单条查询写例外。特征文件 SHA-256 为 `361721CDF5D6098BB87D000E134DF17F1CCCB2498089FD20098EDA4D17B6B2D6`。

## 方法依据

- 混合检索与 RRF：[Elastic hybrid search](https://www.elastic.co/docs/solutions/search/hybrid-search)
- 先召回再精排：[Sentence Transformers retrieve & rerank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
- 多条件必须匹配：[Elastic bool / minimum_should_match](https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-bool-query)
- 高效细粒度匹配：[ColBERT](https://people.eecs.berkeley.edu/~matei/papers/2020/sigir_colbert.pdf)
- 强无答案样本：[SQuAD 2.0](https://nlp.stanford.edu/pubs/rajpurkar2018squad.pdf)
- 选择性预测与风险覆盖：[SelectiveNet](https://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf)
- 防止评测泄漏：[scikit-learn common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)、[GroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- 池化标注及其边界：[NIST TREC](https://trec.nist.gov/howto.html)、[pooling bias](https://www.nist.gov/publications/bias-and-limits-pooling-large-collections)

这些资料用于确定方法方向；本项目的具体门槛仍以本地冻结协议和后续独立校准集为准。

## 验证与产物

- 全量自动化测试：227 项全部通过。
- V16 配置 SHA-256：`10F16F2ECB736B2555F09C79F0DB4F9EA4DEB681030081365C417D115532DCD1`
- Blind V1 回归 SHA-256：`C802A7A5907A571270175FD9A4E9F5AE2C92F2D0F7D2139CA77E09B7C35BE0F3`
- V16 配置：`outputs/evaluation/library_retrieval/v16/selected_retrieval_config_v16.json`
- 历史回归：`outputs/evaluation/library_retrieval/v16/blind_v1_regression.json`
- 回归协议：`data/evaluation/v16/blind_v1_regression_protocol.json`
- 门控特征：`data/evaluation/v16/development/v16_route_gate_features.csv`
- 排名对照：`data/evaluation/v16/development/v16_method_comparison_features.csv`

## 明天继续

1. 新建与现有查询不重叠、按路线均衡的校准集和独立留出集，人与 AI 的标签来源分开记录。
2. 为复合查询增加“必要属性覆盖”特征，并在校准集上学习或选择门槛。
3. 优化精确引文扫描和视觉冷启动，分别报告冷、热延迟。
4. 完成主站真实点击验收，再冻结全新的独立盲测协议；冻结后不再改查询、标签或门槛。
