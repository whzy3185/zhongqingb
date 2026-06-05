# 可复现性说明

## 环境

- Python：见本机 `python --version`
- 依赖：`requirements.txt`
- 随机种子：`20260605`

## 一键复现

```powershell
cd D:\zhongqingb\B_aigc_quality_optimization
python scripts\19_run_all.py
```

## 单步复现顺序

```powershell
python scripts\00_audit_inputs.py
python scripts\01_read_problem.py
python scripts\13_extract_format_requirements.py
python scripts\02_extract_materials.py
python scripts\03_build_features.py
python scripts\04_build_quality_indicator_system.py
python scripts\05_build_weights.py
python scripts\06_quality_evaluation.py
python scripts\07_validity_corrected_score.py
python scripts\08_parameter_effect_analysis.py
python scripts\09_parameter_optimization.py
python scripts\10_multi_agent_evaluation.py
python scripts\11_generate_figures.py
python scripts\12_generate_modeling_paper.py
python scripts\14_build_excel_tables_word.py
python scripts\15_apply_word_template_format.py
python scripts\16_prepare_submission.py
python scripts\17_final_quality_audit.py
```

## 不可自动确认项

- 待人工确认格式项数量：3
- 真实队伍编号需要人工替换；
- 原始提示词、专家标签和真实生成参数未在附件中发现。
