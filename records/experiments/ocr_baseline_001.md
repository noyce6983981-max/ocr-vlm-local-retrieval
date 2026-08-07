# OCR Baseline 001

日期：2026-07-27

## 配置

- 输入：`data/raw/pilot/test.jpg`
- 文字检测：PP-OCRv5 mobile det
- 文字识别：PP-OCRv5 mobile rec
- 推理框架：PaddlePaddle 3.2.2
- 推理设备：RTX 4060 Laptop GPU
- 文档方向、图像矫正和文本行方向：关闭

## 结果

- 识别文本：`蛇年皆顺`
- 识别置信度：约`0.9162`
- 结果JSON：`outputs/ocr/test_res.json`
- 记录截图：`records/screenshots/ocr_baseline_001.jpg`

## 当前结论

首次OCR基线已跑通，检测框覆盖完整竖排文字区域。单样本结果不能代表整体性能，后续需要加入横排、低清晰度、复杂背景和多文本框样本。
