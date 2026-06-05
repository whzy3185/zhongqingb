# 参数-质量关系分析

## 适用性审计

题面 PDF 和附件未提供真实生成参数记录，无法构造 seed、steps、guidance scale、模型名、采样器等参数变量。因此本节不做真实参数优化或参数回归，只做可观测代理特征与模型综合质量分的内部敏感性分析。

## 可观测代理特征相关性

| feature                  | feature_type                              |   spearman_rho |   spearman_p |   kendall_tau |   kendall_p |
|:-------------------------|:------------------------------------------|---------------:|-------------:|--------------:|------------:|
| color_richness_score     | observable_proxy_not_generation_parameter |      0.761905  |    0.0280049 |     0.642857  |   0.0311508 |
| file_size_mb             | observable_proxy_not_generation_parameter |      0.452381  |    0.260405  |     0.285714  |   0.39876   |
| megapixels               | observable_proxy_not_generation_parameter |     -0.436436  |    0.279658  |    -0.267261  |   0.389749  |
| edge_density             | observable_proxy_not_generation_parameter |      0.380952  |    0.351813  |     0.285714  |   0.39876   |
| aspect_ratio             | observable_proxy_not_generation_parameter |     -0.354604  |    0.388751  |    -0.267261  |   0.389749  |
| artifact_risk_score      | observable_proxy_not_generation_parameter |     -0.238095  |    0.570156  |    -0.214286  |   0.548413  |
| proxy_prompt_token_count | observable_proxy_not_generation_parameter |      0.218218  |    0.603645  |     0.157243  |   0.621873  |
| sharpness_score          | observable_proxy_not_generation_parameter |      0.0239525 |    0.955106  |    -0.0363696 |   0.90078   |

## 建模限制

这些相关性只能说明本批附件中“可观测图像属性”和模型分数的关系，不能解释为生成参数的因果影响。
