# 已确认命令

以下命令均在项目根目录 PowerShell 中运行。

## 使用项目Python

```powershell
.\.venv\Scripts\python.exe --version
```

## 安装PyTorch

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 -i https://pypi.tuna.tsinghua.edu.cn/simple -f https://mirrors.aliyun.com/pytorch-wheels/cu128/
```

## 验证PyTorch GPU

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 安装PaddlePaddle

```powershell
.\.venv\Scripts\python.exe -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/
```

## 验证PaddlePaddle GPU

```powershell
.\.venv\Scripts\python.exe -c "import paddle; paddle.utils.run_check(); print(paddle.device.get_device())"
```

## 安装PaddleOCR

```powershell
.\.venv-paddle\Scripts\python.exe -m pip install paddleocr==3.7.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 运行OCR基线

```powershell
.\.venv-paddle\Scripts\python.exe scripts\run_ocr.py data\raw\pilot\test.jpg
```

## 运行正式批量OCR

```powershell
.\.venv-paddle\Scripts\python.exe scripts\batch_ocr.py --force
```

结果：

- `outputs/ocr_batch/summary.csv`
- `outputs/ocr_batch/failures.jsonl`
- `outputs/ocr_batch/json/`
- `outputs/ocr_batch/visualizations/`

## 构建OCR文本语料

```powershell
.\.venv-paddle\Scripts\python.exe scripts\build_text_corpus.py
```

## 下载BGE-M3（ModelScope）

```powershell
.\.venv\Scripts\python.exe scripts\download_bge_m3.py
```

## 构建FAISS文本索引

```powershell
.\.venv\Scripts\python.exe scripts\build_text_index.py
```

## 自然语言检索

```powershell
.\.venv\Scripts\python.exe scripts\search_text.py "哪张图片介绍了卷积层、池化层和全连接层？" --top-k 3
```

## 评测文本检索

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_text_retrieval.py
```

## 安装视觉检索环境

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_vl_env.ps1
```

## 下载Qwen3-VL-Embedding-2B

```powershell
.\.venv-vl\Scripts\python.exe scripts\download_qwen3_vl_embedding.py
```

## 获取锁定版本的Qwen3-VL官方代码

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\fetch_qwen3_vl_code.ps1
```

## 构建视觉索引

```powershell
.\.venv-vl\Scripts\python.exe scripts\build_visual_index.py
```

## 评测视觉检索

```powershell
.\.venv-vl\Scripts\python.exe scripts\evaluate_visual_retrieval.py
```

## 评测多模态融合

```powershell
.\.venv-paddle\Scripts\python.exe scripts\evaluate_fusion.py
```

## 构建24张dataset_v1

```powershell
.\.venv-paddle\Scripts\python.exe scripts\build_dataset_v1.py
```

## dataset_v1完整评测

```powershell
.\.venv-paddle\Scripts\python.exe scripts\batch_ocr.py --manifest data\manifest\dataset_v1_manifest.jsonl --output outputs\ocr_dataset_v1 --device gpu --force

.\.venv\Scripts\python.exe scripts\build_text_corpus.py --manifest data\manifest\dataset_v1_manifest.jsonl --ocr-dir outputs\ocr_dataset_v1\json --output data\processed\dataset_v1_corpus.jsonl --summary outputs\text_corpus_dataset_v1\summary.json

.\.venv\Scripts\python.exe scripts\build_text_index.py --corpus data\processed\dataset_v1_corpus.jsonl --output artifacts\text_index_dataset_v1

.\.venv\Scripts\python.exe scripts\evaluate_text_retrieval.py --queries data\evaluation\dataset_v1_queries.csv --manifest data\manifest\dataset_v1_manifest.jsonl --index-dir artifacts\text_index_dataset_v1 --output outputs\evaluation_dataset_v1\text_retrieval.json

.\.venv-vl\Scripts\python.exe scripts\build_visual_index.py --manifest data\manifest\dataset_v1_manifest.jsonl --output artifacts\visual_index_dataset_v1

.\.venv-vl\Scripts\python.exe scripts\evaluate_visual_retrieval.py --queries data\evaluation\dataset_v1_queries.csv --index-dir artifacts\visual_index_dataset_v1 --output outputs\evaluation_dataset_v1\visual_retrieval.json

.\.venv\Scripts\python.exe scripts\evaluate_fusion.py --queries data\evaluation\dataset_v1_queries.csv --ocr-summary outputs\ocr_dataset_v1\summary.csv --text-scores outputs\evaluation_dataset_v1\text_score_matrix.npz --visual-scores outputs\evaluation_dataset_v1\visual_score_matrix.npz --output outputs\evaluation_dataset_v1\fusion_retrieval.json
```

## 启动Streamlit实验结果浏览器

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ui.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## 命令行运行任意自然语言实时检索

```powershell
.\.venv\Scripts\python.exe scripts\live_search.py "哪张图片展示海边日落和水面倒影？"
```

首次运行会顺序调用`.venv`中的BGE-M3和`.venv-vl`中的Qwen3-VL；相同问题再次执行时读取`outputs/live_cache/`。

## 独立盲测

日常采集请在8501主站选择“独立盲测”；冻结和人工相关性审核请在8502选择“独立盲测审核”。冻结后也可以用命令行断点生成候选：

```powershell
.\.venv\Scripts\python.exe scripts\build_blind_study_candidates.py --study-dir data\evaluation\blind_study_v1 --library-dir outputs\user_library --limit 1
```

人工审核全部完成并由页面生成正式文件后，运行不可调参的正式评测：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_live_search_blind.py --protocol data\evaluation\blind_study_v1\evaluation_protocol.json
```

## V16 历史回归与门控特征

Blind V1 已参与过 V16 分析，以下命令只用于历史回归，不是新的独立盲测：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_live_search_blind.py --protocol data\evaluation\v16\blind_v1_regression_protocol.json

.\.venv\Scripts\python.exe scripts\capture_route_gate_features.py --queries data\evaluation\public_dataset_1500_retrieval_queries_formal.csv --output data\evaluation\v16\development\legacy99_v16_final_features.csv --expected-policy 16
```

V16 选定配置位于 `outputs/evaluation/library_retrieval/v16/selected_retrieval_config_v16.json`。全新的路线均衡校准集与独立留出集需另建目录，不能覆盖 Blind V1 或 legacy99。

## V16 路线均衡校准与独立留出

查询生成只读取来源元数据和 OCR，不调用检索：

```powershell
.\.venv\Scripts\python.exe scripts\build_v16_balanced_queries.py
```

批量预热使用与产品完全相同的组件缓存格式，避免逐题重复加载模型：

```powershell
.\.venv\Scripts\python.exe scripts\score_library_text_batch.py --queries data\evaluation\v16\calibration\frozen_queries.csv --splits calibration --required-review-status frozen_ai_source_verified --output outputs\evaluation\library_retrieval\v16\calibration_text_bm25_scores.npz --live-cache-dir outputs\live_cache\components

.\.venv-vl\Scripts\python.exe scripts\score_library_visual_batch.py --queries data\evaluation\v16\calibration\frozen_queries.csv --splits calibration --required-review-status frozen_ai_source_verified --product-required-only --output outputs\evaluation\library_retrieval\v16\calibration_visual_scores.npz --live-cache-dir outputs\live_cache\components
```

校准评测可以复现；独立留出已经完成首次正式运行，不得用于继续调整 V16：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_live_search_blind.py --protocol data\evaluation\v16\calibration\evaluation_protocol.json

# 只用于复核已封存报告，任何后续改动均必须进入新版本与新留出集
.\.venv\Scripts\python.exe scripts\evaluate_live_search_blind.py --protocol data\evaluation\v16\holdout\evaluation_protocol.json
```

## 命令行增量加入一张图片

```powershell
.\.venv\Scripts\python.exe scripts\ingest_image.py "D:\图片\示例.png" --display-name "示例资料" --category personal_document
```

也可以启动Streamlit后选择“上传并入库”。个人原图、OCR结果和增量索引保存在`outputs/user_library/`，默认不提交Git。

## 批量导入多格式资料

推荐将大量图片放进一个ZIP，也可直接同时传入PDF、DOCX和PPTX：

```powershell
.\.venv\Scripts\python.exe scripts\batch_ingest.py "D:\资料\images.zip" "D:\资料\report.pdf" "D:\资料\slides.pptx"
```

程序会把文档转换为页面图，整批只加载一次OCR、文本和视觉模型。

## 下载并测试Qwen3-VL-Reranker-2B

```powershell
.\.venv-vl\Scripts\python.exe scripts\download_qwen3_vl_reranker.py
.\.venv-vl\Scripts\python.exe scripts\test_vl_reranker.py
```

## 实时检索时启用重排

```powershell
.\.venv\Scripts\python.exe scripts\live_search.py "哪张低清图片介绍文字识别、手写识别和表格识别？" --rerank-top-k 10
```

## 批量评测重排

```powershell
.\.venv-vl\Scripts\python.exe scripts\evaluate_reranker.py
```

## V17 校准集 Top-K 候选验证与 parser 诊断

以下命令只读取校准集；不要把参数改到最终留出集：

```powershell
.\.venv-vl\Scripts\python.exe scripts\score_v17_candidate_verification.py --top-k 5
.\.venv\Scripts\python.exe scripts\calibrate_v17_candidate_gate.py
.\.venv\Scripts\python.exe scripts\evaluate_v17_parser.py
```

K=5 评分按查询原子保存，进程中断后使用相同参数会从已完成前缀继续。
校准脚本从同一份 K=5 评分派生 K=1/3/5，默认把新候选配置写入
`config/v17_candidate_verification_gate_topk_candidate.json`，不会覆盖历史
Top-1 配置，也不会打开留出集。

## V17 最终方法锁只读预检

以下命令会验证方法代码、提示词、4.25GB 本地模型目录、冻结查询与校准产物，
但不会打开或执行最终留出集：

```powershell
$runtimeRoot = "D:\private-v17-runtime"
.\.venv\Scripts\python.exe scripts\run_v17_holdout_once.py --runtime-root $runtimeRoot
```

正常的当前结果应为 `blocked`，唯一阻塞项是尚无独立人工审核授权。不要添加
`--execute`；只有两名审核者独立完成全部候选盲审、第三人完成冲突裁决，且授权
文件绑定全部输入哈希后，才允许进行唯一一次最终评测。receipt 创建后即使评测
进程失败也禁止重跑，任何方法变化必须升级到 V18。

## 重新采集并审计200页公开候选集

```powershell
.\.venv\Scripts\python.exe scripts\collect_public_dataset_200.py
.\.venv\Scripts\python.exe scripts\audit_public_dataset_200.py
.\.venv\Scripts\python.exe scripts\build_dataset_contact_sheets.py
```

原始图片保存在`data/incoming/200页全部放这里/`，默认不提交Git；逐页来源和许可证保存在`data/evaluation/public_dataset_200_sources.csv`。

## 200页元数据与防泄漏划分

```powershell
# 将公开来源、许可证、类别和复核标记回填到已入库页面
.\.venv\Scripts\python.exe scripts\apply_public_dataset_metadata.py

# 生成按视觉近重复组隔离的120/40/40划分
.\.venv\Scripts\python.exe scripts\build_public_dataset_splits.py

# 生成200页OCR质量特征和30页重点复核队列
.\.venv\Scripts\python.exe scripts\analyze_public_dataset_quality.py

# 仅对低置信度含文字页做四方向OCR对照
.\.venv-paddle\Scripts\python.exe scripts\evaluate_selective_rotation.py

# 安装保守接受的文本覆盖层并重建BGE-M3索引
.\.venv\Scripts\python.exe scripts\apply_rotation_overrides.py
.\.venv\Scripts\python.exe scripts\rebuild_user_text_index.py

# 生成候选查询并建立30条人工复核队列
.\.venv\Scripts\python.exe scripts\generate_public_dataset_queries.py
.\.venv\Scripts\python.exe scripts\prioritize_query_review.py
.\.venv\Scripts\python.exe scripts\build_query_review_sheets.py

# 学习式门控候选实验
.\.venv\Scripts\python.exe scripts\train_learned_gate.py
.\.venv\Scripts\python.exe scripts\train_preference_gate.py
```
