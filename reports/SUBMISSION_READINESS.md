# 提交就绪检查

## 当前可用文件

- 占位命名论文 PDF：`submission/B题论文_AI生成内容质量评估与参数优化.pdf`
- 占位命名论文 Word：`submission/B题论文_AI生成内容质量评估与参数优化.docx`
- 占位命名支撑材料：`submission/B题支撑材料_zhongqingB.zip`
- AI 工具使用详情说明：`submission/AI工具使用详情说明.md`

## 正式提交前必须执行

将 `B真实队伍编号` 替换为报名页面显示的真实队伍编号：

```powershell
cd D:\zhongqingb\B_aigc_quality_optimization
python scripts\18_rename_submission.py B真实队伍编号
```

运行后应提交：

- `submission/B真实队伍编号.pdf`
- `submission/B真实队伍编号材料.zip`

## 当前不能直接提交的内容

- 首页仍含 `待替换` 的通用 Word/PDF 不能作为最终命名文件直接提交；
- 任何测试编号文件不能提交，除非测试编号正是真实队伍编号；
- 原始赛题附件一般不需要打入支撑材料，支撑材料包主要包含程序、结果表、图和报告。

## 人工复核清单

1. 队伍编号正确；
2. PDF 文件名符合 `B+队伍编号.pdf`；
3. 支撑材料文件名符合 `B+队伍编号+材料.zip`；
4. 论文不含学校和队员姓名；
5. 图题、表题和公式编号符合人工确认的最终格式；
6. AI 工具使用详情说明随支撑材料一并提交；
7. 当前 PDF 页数为 10 页，未超过 20 页目标。
