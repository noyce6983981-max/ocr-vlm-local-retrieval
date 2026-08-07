# OCR正式批处理基线 001

日期：2026-07-29

## 目的

将单图OCR升级为可复现的manifest批处理实验，为后续文本检索和多模态改进提供结构化输入。

## 配置

- 数据：12张异构试运行图片
- manifest：`data/manifest/pilot_manifest.jsonl`
- 模型：PP-OCRv5 mobile det + PP-OCRv5 mobile rec
- 设备：RTX 4060 Laptop GPU，`gpu:0`
- 文字方向、文档矫正和方向分类：关闭
- 执行命令：

```powershell
.\.venv-paddle\Scripts\python.exe scripts\batch_ocr.py --force
```

## 真实结果

- 输入样本：12
- 无程序异常：12/12
- JSON结果：12
- 可视化结果：12
- 失败日志条数：0
- 逐图推理与结果保存累计时间：23.585秒
- 11张有OCR文字图片的图像级平均置信度均值：0.8863

## 关键样本

| 样本 | 类型 | 文本框 | 字符数 | 平均置信度 |
|---|---|---:|---:|---:|
| pilot_005_paper_page_01 | 论文 | 50 | 1328 | 0.9913 |
| pilot_007_campus_photo | 自然图像 | 0 | 0 | 0 |
| pilot_009_lowres_ocr_poster | 低清文字图 | 9 | 56 | 0.5275 |

## 初步结论

- 清晰论文页面的OCR置信度高，可直接作为文本检索基线的输入。
- 无文字自然图像返回空文本，纯OCR检索无法覆盖该类输入。
- 低清文字图置信度最低且存在明显错字，后续需要质量门控或视觉语义分支。
- 置信度不是准确率；正式OCR精度仍需人工真值和CER评价。

## 本人待完成

- 审核12条manifest标签。
- 人工检查所有可视化结果并记录错误类型。
- 选择代表性成功、失败和边界案例用于README。
