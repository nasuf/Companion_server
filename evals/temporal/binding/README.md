# TIME02D 日期绑定回归

12 个不同的合成场景：7 个跨事件、重复日期、否定、假设、引用和改期目标，5 个正常召回对照。
与实际回复执行器共用真实提示词渲染、模型及回复处理；用户、记忆、路由和外部业务 IO 为合成夹具。
这不是盲验、完整时间质量资格、生产发送或存储 E2E。

对同一份生产提示词快照、同一模型分别指定旧版和候选源码，运行：

```sh
python evals/temporal_binding/run.py --source-root /isolated/checkout \
  --prompt-snapshot /private/prompts.json --output /private/fresh-result.json \
  --model doubao-seed-character-260628 --samples 5 --live
```

仅从环境提供 ARK / DASHSCOPE 凭据，不放入源码或报告。源码必须没有 `.env`。
输出包含私有的渲染提示词及答复，应限制访问。所有失败和裁判未知保留在固定分母；
中断的报告不得删除或冒充完成，重跑使用新的报告名并记录原因。
确定性日期真值及检索完整性以 `tests/test_temporal_binding.py` 为准，
自动裁判结果需逐条复核，不能用回复评分代替原始证据和完整链路验收。
