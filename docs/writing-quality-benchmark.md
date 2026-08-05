# 写作质量基准

`readraft-writing-benchmark` 用固定的原创测试场景比较 Low、Standard、Max，记录
输出哈希、约束命中、连续性、视角泄漏、重复、模型痕迹、延迟和 token。自动分只覆盖
可以机械核对的部分；可读性、人物能动性、场景推进、具体性和对话仍通过盲评表评价。

离线自检（使用仓库内固定基线文本）：

```bash
readraft-writing-benchmark --output benchmark-results/baseline
```

评价已经保存的模型输出：

```bash
readraft-writing-benchmark \
  --responses benchmark-inputs \
  --output benchmark-results/run-001
```

响应文件命名为 `<case_id>.<low|standard|max>.txt`。可选的同名 `.json` 记录
`latency_seconds`、`input_tokens`、`output_tokens`、`provider` 和 `model`。

实时比较使用环境变量，不把 Key 放进命令历史：

```bash
export READRAFT_BENCH_FAST_API_KEY='...'
export READRAFT_BENCH_FAST_PROVIDER='deepseek'
export READRAFT_BENCH_FAST_MODEL='deepseek-v4-flash'
export READRAFT_BENCH_FAST_BASE_URL='https://api.deepseek.com'
export READRAFT_BENCH_QUALITY_API_KEY='...'
export READRAFT_BENCH_QUALITY_PROVIDER='deepseek'
export READRAFT_BENCH_QUALITY_MODEL='deepseek-v4-pro'
export READRAFT_BENCH_QUALITY_BASE_URL='https://api.deepseek.com'
readraft-writing-benchmark --live --output benchmark-results/live-001
```

每次运行生成：

- `report.json`：机器可比较的完整记录；
- `report.md` / `report.html`：模式汇总；
- `blind-review.csv`：人工评分表；
- `S001.txt` 等：打乱模式信息后的盲评样本；
- `blind-key.json`：盲评结束后再打开的样本映射。

不同提交之间只有在用例集哈希、模型、模式和响应策略一致时才直接比较。真实回归门槛应
同时要求“自动分不下降、硬失败不增加、盲评不下降”，不能只优化字数或一个总分。
