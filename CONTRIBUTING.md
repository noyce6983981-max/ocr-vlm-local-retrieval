# 参与开发

## 本地验证

项目使用 Python 3.11。无需下载模型即可运行 CPU 单元测试：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-ci.txt
.\.venv\Scripts\python.exe -m pytest -q
```

涉及 OCR、文本向量、视觉向量或视觉重排的改动，还应按照 `records/COMMANDS.md` 在对应隔离环境中运行相关回归。

## 数据与隐私

- 不提交原始个人文档、OCR 全文、模型审核包、查询缓存、模型权重或向量索引。
- 测试样例应使用虚构信息；真实数据只能保存在 Git 忽略的本地目录。
- 新增公开数据必须记录来源、作者和许可，并遵守 `data/DATA_POLICY.md`。
- 改动检索门槛或排序策略时，必须新建校准版本；不得根据已解封留出集继续调参。

## 提交建议

一次提交只处理一个可说明的目标，并同时提交对应测试或实验记录。提交前运行完整单元测试与公开发行安全审计。
