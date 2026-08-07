# 安全与隐私

本项目默认在本地处理文档。原始图片、OCR 全文、索引、模型权重、查询缓存和人工审核包不属于公开仓库内容。

## 报告问题

如果发现代码执行、压缩包路径穿越、文件隔离、隐私泄露或依赖安全问题，请不要在公开 Issue 中粘贴原始文档、OCR 全文、密钥或个人信息。请仅描述复现条件，并在仓库启用私密安全报告后使用 GitHub Private Vulnerability Reporting。

## 公开前检查

公开发行副本必须通过：

```powershell
python scripts/check_public_release.py .
python -m pytest -q
```

安全检查会拒绝常见密钥、个人 Windows 绝对路径、敏感评测文件以及数据/记录目录中的身份证号、手机号和邮箱样式内容。测试代码中用于验证隐私规则的虚构样例不参与 PII 扫描。
