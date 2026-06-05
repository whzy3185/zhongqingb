# WORK_CONTINUATION

## 已完成内容

已完成工作区创建、RAR 解压、题面与细则读取、模板转换尝试、附件解析、特征提取、图像质量评价、附件内部相对高/中/低等级、视频时序质量评价、图表、论文、Word/PDF、支撑材料 zip 和最终审计。

## 当前主模型

主模型为“无参考图像质量评价 + 组合赋权综合评价 + 光流时序质量评价”。参数优化不是题面主问题，且附件无真实参数，已按不适用处理。

## 当前主论文

- `paper/main.md`
- `paper/main_with_excel_tables.docx`
- `paper/main_with_excel_tables.pdf`

## 当前推荐提交文件

- `submission/B题论文_AI生成内容质量评估与参数优化.pdf`
- `submission/B题支撑材料_zhongqingB.zip`
- `submission/AI工具使用详情说明.md`
- 上述文件为占位命名；正式提交时以 `scripts\18_rename_submission.py` 生成的 `B真实队伍编号.pdf` 与 `B真实队伍编号材料.zip` 为准。

## 一键复现命令

```powershell
cd D:\zhongqingb\B_aigc_quality_optimization
python scripts\19_run_all.py
```

## 格式状态

参赛细则已读取；旧 `.doc` 模板已尝试转换。当前 PDF 页数：11。

## 是否读取模板

是，已转换为 data/extracted/competition_template_converted.docx

## 是否有待人工确认格式项

是，数量：3。

## 是否需要替换真实队伍编号

是。当前通用论文首页仍为 `队伍编号：待替换`。若提交目录中存在 `B202600001.pdf/docx`，它只是脚本验证样例；除非这正是真实队伍编号，否则不能直接提交。

## 是否需要压缩页数

否，当前程序统计不超过 20 页。

## 后续人工工作

1. 替换真实队伍编号。
2. 运行 `python scripts\18_rename_submission.py B真实队伍编号`。
3. 人工复核公式排版、图表题注、表题格式和页边距。
4. 人工确认没有参赛队员与学校信息。
5. 人工确认语义代理描述和内容类型分类是否需要调整。

## 如果需要继续优化

从 `reports/FINAL_QUALITY_AUDIT.md` 和 `reports/FORMAT_REQUIREMENTS.md` 的 `MANUAL_CHECK_REQUIRED` 项开始。
