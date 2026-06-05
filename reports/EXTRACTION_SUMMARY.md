# 材料解析摘要

## 样本结构

| sample_id   | sample_type   | content_type   | source_path                                                          | file_name    | original_prompt_available   | quality_label_available   |
|:------------|:--------------|:---------------|:---------------------------------------------------------------------|:-------------|:----------------------------|:--------------------------|
| image_01    | image         | 写实风景       | D:\zhongqingb\B题：AI生成内容的质量评估与参数优化\附件1\1.png        | 1.png        | False                       | False                     |
| image_02    | image         | 人物肖像       | D:\zhongqingb\B题：AI生成内容的质量评估与参数优化\附件1\2.png        | 2.png        | False                       | False                     |
| image_03    | image         | 艺术插画       | D:\zhongqingb\B题：AI生成内容的质量评估与参数优化\附件1\3.png        | 3.png        | False                       | False                     |
| image_04    | image         | 动态街景       | D:\zhongqingb\B题：AI生成内容的质量评估与参数优化\附件1\4.png        | 4.png        | False                       | False                     |
| image_05    | image         | 写实风景       | D:\zhongqingb\B题：AI生成内容的质量评估与参数优化\附件1\5.jpg        | 5.jpg        | False                       | False                     |
| image_06    | image         | 人物插画       | D:\zhongqingb\B题：AI生成内容的质量评估与参数优化\附件1\6.jpg        | 6.jpg        | False                       | False                     |
| image_07    | image         | 水墨插画       | D:\zhongqingb\B题：AI生成内容的质量评估与参数优化\附件1\7.jpg        | 7.jpg        | False                       | False                     |
| image_08    | image         | 像素街景       | D:\zhongqingb\B题：AI生成内容的质量评估与参数优化\附件1\8.jpg        | 8.jpg        | False                       | False                     |
| video_01    | video         | 车流视频       | D:\zhongqingb\B题：AI生成内容的质量评估与参数优化\附件2\车流视频.mp4 | 车流视频.mp4 | False                       | False                     |

## 元数据

| sample_id   | file_name    |   width |   height | mode   | format   | dpi              |   exif_items | metadata_keys                                |   size_bytes |   readable |   frame_count |   fps |   duration_sec |
|:------------|:-------------|--------:|---------:|:-------|:---------|:-----------------|-------------:|:---------------------------------------------|-------------:|-----------:|--------------:|------:|---------------:|
| image_01    | 1.png        |    2730 |     1535 | RGBA   | PNG      | (96.012, 96.012) |            0 | dpi                                          |      5486221 |        nan |           nan |   nan |      nan       |
| image_02    | 2.png        |    1773 |     2364 | RGBA   | PNG      | (96.012, 96.012) |            0 | dpi                                          |      5044845 |        nan |           nan |   nan |      nan       |
| image_03    | 3.png        |    2048 |     2048 | RGBA   | PNG      | (96.012, 96.012) |            0 | dpi                                          |      4421390 |        nan |           nan |   nan |      nan       |
| image_04    | 4.png        |    1536 |     2730 | RGBA   | PNG      | (96.012, 96.012) |            0 | dpi                                          |      5620743 |        nan |           nan |   nan |      nan       |
| image_05    | 5.jpg        |    2048 |     2048 | RGB    | JPEG     | (120, 120)       |            0 | dpi,jfif,jfif_density,jfif_unit,jfif_version |      1039345 |        nan |           nan |   nan |      nan       |
| image_06    | 6.jpg        |    2048 |     2048 | RGB    | JPEG     | (120, 120)       |            0 | dpi,jfif,jfif_density,jfif_unit,jfif_version |       459828 |        nan |           nan |   nan |      nan       |
| image_07    | 7.jpg        |    2048 |     2048 | RGB    | JPEG     | (120, 120)       |            0 | dpi,jfif,jfif_density,jfif_unit,jfif_version |       544822 |        nan |           nan |   nan |      nan       |
| image_08    | 8.jpg        |    2048 |     2048 | RGB    | JPEG     | (120, 120)       |            0 | dpi,jfif,jfif_density,jfif_unit,jfif_version |       298720 |        nan |           nan |   nan |      nan       |
| video_01    | 车流视频.mp4 |    1920 |     1080 | nan    | nan      | nan              |          nan | nan                                          |      6691691 |          1 |           121 |    24 |        5.04167 |

## 提示词与标签状态

- 原始生成提示词：未在 PDF、文件名、EXIF 或 PNG 元数据中发现。
- 专家高/中/低质量标签：未发现。
- 已建立 `data/labels/proxy_prompts.csv`，其中的描述仅为可视内容结构化代理，用于计算题面要求的语义要素覆盖代理。
- 附件 1 共 8 张图片，附件 2 共 1 个视频。
- 可视审计未发现明确产品渲染样本；本工程不补造该类型，按附件实际图像类型进行敏感性分析。
