# 独立盲测 V1 协议

日期：2026-08-01

## 当前轮次状态

用户最初提交的40条通用网络搜索词已归档到`data/evaluation/archive/blind_study_v1_user_supplied_generic_40.csv`。这些词多数依赖天气、票务、地图、挂号等外部实时服务，不适合作为本地图片/文档检索的主要评测题。

按用户授权，当前采集文件已替换为60条助手生成的本地资料库诊断查询：50条常规检索、10条开放集探针。协议中的`query_origin`为`assistant_generated`，`study_type`为`assistant_generated_diagnostic`。这轮可以用于发现召回、排序和拒答问题，但不得表述为“独立用户盲测”；真正的独立成绩仍需另一位查询作者在看不到系统结果的条件下提供题目。

用户确认进入下一步后，本轮于`2026-08-01T10:59:37Z`冻结。查询指纹为`cc5904be41605a7a3c6b9344622b79e6c525b09e8fd79cd2a6f3a77f4d8ff598`，绑定资料库revision为`3051b34cf6e4`。候选从`11:00:15Z`生成至`11:29:02Z`：60/60条查询均有18个唯一候选，所有候选ID存在于manifest，智能混合、OCR文字、视觉语义三路均有覆盖；策略锁定为V15、配置revision为`523a8382715f`，审核顺序全部通过`sha256_blinded_v1`复算。候选文件约254KB。当前状态为`reviewing`，尚未产生人工相关性标签或正式成绩。

## 目标

建立一轮真正独立于当前检索结果的新测试：查询先由用户按真实需求自然提出，采集阶段不运行检索也不展示任何结果；达到目标后冻结查询与资料库版本，随后才建立候选池、做人工相关性判断并运行正式评测。

默认采集60条，最终有效查询不得少于50条，最多100条；其中至少10条由作者明确作为“可能库中无答案”的开放集探针。作者的答案预期只用于样本平衡，不是真值，并且在人工相关性审核页隐藏。

## 参考方法与本项目映射

- [NIST TREC](https://trec.nist.gov/howto.html)把训练数据与最终评测数据分开，并用多个检索run的Top-N组成待判断池。本项目对应为“先冻结、后检索”，再合并智能混合、OCR文字和视觉语义三路Top-12。
- [TREC relevance judgments](https://trec.nist.gov/data/reljudge_eng.html)强调qrels必须与固定文档集合匹配，并通过pooling提高相关文档覆盖。本项目把资料库revision写入协议，内容一旦变化即停止采集或审核。
- [MS MARCO](https://www.microsoft.com/en-us/research/publication/ms-marco-human-generated-machine-reading-comprehension-dataset/)使用真实匿名用户查询和人工答案。本项目不由系统生成题目，也不在输入框提供可照抄的具体示例。
- [BEIR](https://arxiv.org/abs/2104.08663)强调跨任务、跨领域和零样本泛化。本轮不把旧回归题混入新集合，正式报告会按实际查询路线拆分结果，但不根据盲测成绩调权重。

## 状态机

1. `collecting`：主站只接受查询和作者预期；禁止调用检索、查看候选或浏览系统接收决定。
2. `frozen`：写出`frozen_queries.csv`和规范化SHA-256；查询、顺序与绑定资料库revision不可再改。
3. `reviewing`：三路检索形成最多18项的去重候选池。候选按“冻结指纹 + query_id + item_id”确定性盲排，审核页不显示模型分数、原始排名、候选来源、系统接收决定或作者预期。
4. `completed`：所有冻结查询均有人工作出`answerable`、`no_answer`或`excluded`裁决；排除后少于50条时拒绝生成正式集。

## 防泄漏与复现闸门

- 查询采集函数与候选生成函数分离；采集只原子写CSV。
- 协议绑定资料库ID和内容revision，资料变化即硬失败。
- 冻结快照使用规范化SHA-256；任何手工改写都会阻止候选生成。
- 首条候选锁定搜索策略版本、检索配置revision、资料库revision与pool方法；后续不允许混用。
- 候选逐题落盘，支持断点续跑；大型实时检索payload使用临时目录并在每题结束后删除。
- qrels导出后同时生成`evaluation_protocol.json`，锁定query fingerprint、策略版本、查询数和输出位置。
- 正式运行期间禁止修改查询、qrels、路由、权重或阈值；发现问题进入下一版本，不回写本轮成绩。

## 人工审核规则

- `answerable`：至少选择一张相关图片；多张同样满足需求时全部勾选。
- `no_answer`：确认资料库没有满足问题的页面，不得填写相关ID。
- `excluded`：问题含隐私风险、语义含糊、依赖不可见上下文或无法公平判断。
- 候选池漏掉正确图片时可补充资料库中的有效item ID，并记录核对范围。
- 推荐重要查询浏览完整资料库；仅看候选池时必须保留`candidate_pool`证据范围。

## 产物

- `data/evaluation/blind_study_v1/protocol.json`
- `data/evaluation/blind_study_v1/submissions.csv`
- `data/evaluation/blind_study_v1/frozen_queries.csv`
- `data/evaluation/blind_study_v1/candidates.jsonl`
- `data/evaluation/blind_study_v1/reviews.csv`
- `data/evaluation/blind_study_v1/formal_queries.csv`
- `data/evaluation/blind_study_v1/evaluation_protocol.json`
- `outputs/evaluation/library_retrieval/blind_study_v1_formal.json`

## 已知限制

这是一套适合本地单用户项目的严谨实用流程，不等于完整TREC。候选池只有当前系统的三条检索路线，仍可能漏掉三路都未召回的相关文档；单人审核也无法测量标注者一致性。因此正式报告必须同时披露pooling偏差、审核范围和查询数量，不能把结果包装成通用SOTA结论。

## 模型辅助相关性审核

为减少用户逐条阅读英文或低清图片的负担，本轮在候选池冻结后增加了模型辅助审核。审核包只包含查询文字、候选图片、候选ID、来源文件名和OCR文字；不包含系统排名、分数、检索路线、候选来源、系统接收决定或作者预期。安全审核包SHA-256为`fb606ecea032f1459d22de6656a8f473c4c27343c8155f2bff701bdc650f4161`。

60条查询先分为三个互不重叠的20题批次，交给三位无记忆审核者。50条高置信、无不确定候选的判断以`model_single_high`写入；其余10条交给两位新的无记忆审核者复核，只有裁决和完整相关ID集合精确一致、双方置信度均不低于0.8且没有不确定ID时，才以`model_consensus`写入。第二轮仍有4条分歧，再交给第三位新审核者复核，其中1条形成多数精确共识。

截至页面验收，正式审核表已有57/60条：50条`model_single_high`、7条`model_consensus`。剩余三条`blind_v1_045`、`blind_v1_046`、`blind_v1_049`因三位无记忆审核者仍未对完整相关集合形成一致，保留给用户最终裁决。治理页面会显示各审核者的中文理由、置信度和不确定图片ID，但继续隐藏系统排名、分数和作者预期。模型辅助标签不得表述为人工标注；三条人工裁决完成前，不生成正式成绩。
