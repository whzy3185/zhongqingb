# 输入审计报告

## 审计结论

- 工作区：`D:\zhongqingb`
- 工程目录：`D:\zhongqingb\B_aigc_quality_optimization`
- 原始材料目录：`D:\zhongqingb\B题：AI生成内容的质量评估与参数优化`
- B 题题面 PDF：PASS
- 参赛细则 PDF：PASS
- 论文模板：PASS
- 诚信参赛告知书：PASS
- 附件图片数：8
- 附件视频数：1

## 文件清单

| name                                                 | suffix   |   size_bytes | pages_or_samples   | readable   | requires_ocr   | manual_required   | notes                                             |
|:-----------------------------------------------------|:---------|-------------:|:-------------------|:-----------|:---------------|:------------------|:--------------------------------------------------|
| 2026年第八届中青杯全国大学生数学建模竞赛参赛细则.pdf | .pdf     |       192834 | 3                  | True       | False          | False             |                                                   |
| 2026年第八届中青杯全国大学生数学建模竞赛论文模板.doc | .doc     |        16471 |                    | True       | False          | True              | Word 模板文件；旧 .doc 需 Word COM 转换或人工复核 |
| B题：AI生成内容的质量评估与参数优化.pdf              | .pdf     |       155836 | 2                  | True       | False          | False             |                                                   |
| 中青杯全国大学生数学建模竞赛诚信参赛告知书.pdf       | .pdf     |       138422 | 1                  | True       | False          | False             |                                                   |
| 1.png                                                | .png     |      5486221 | 2730x1535          | True       | False          | False             |                                                   |
| 2.png                                                | .png     |      5044845 | 1773x2364          | True       | False          | False             |                                                   |
| 3.png                                                | .png     |      4421390 | 2048x2048          | True       | False          | False             |                                                   |
| 4.png                                                | .png     |      5620743 | 1536x2730          | True       | False          | False             |                                                   |
| 5.jpg                                                | .jpg     |      1039345 | 2048x2048          | True       | False          | False             |                                                   |
| 6.jpg                                                | .jpg     |       459828 | 2048x2048          | True       | False          | False             |                                                   |
| 7.jpg                                                | .jpg     |       544822 | 2048x2048          | True       | False          | False             |                                                   |
| 8.jpg                                                | .jpg     |       298720 | 2048x2048          | True       | False          | False             |                                                   |
| 车流视频.mp4                                         | .mp4     |      6691691 | 121 frames         | True       | False          | False             |                                                   |

## 风险与人工确认

- 附件图片无 EXIF/PNG 文本提示词元数据，题面要求的“文本提示词结构化”需要使用可视内容代理描述，不能解释为原始生成提示词。
- 附件 1 共 8 张图，而题面表述“每种类型包含高、中、低三个质量等级”与“四种内容类型”存在数量张数上的潜在歧义，不能伪造专家等级。
- 可视审计未发现明确“产品渲染”样本，因此类型敏感性分析以附件实际内容类型为准，并在论文中说明题面类型覆盖要求与附件实际样本存在差异。
- 论文模板为旧版 `.doc`，若 Word COM 转换失败，则只能依据参赛细则复刻格式并标注待人工确认项。
