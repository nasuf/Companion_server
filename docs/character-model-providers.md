# 角色模型接入与注册

后端通过 `app/services/llm/providers.py` 集中管理供应商、凭据和 API 地址；可选
模型、价格和启用状态继续由后台「系统设置 → 模型库」动态管理。主回复模型与
辅助小模型可使用不同供应商，因此可以只替换聊天角色模型，不影响意图识别、
记忆抽取和情绪分类。

## 1. GitHub Actions 配置

在 `Companion_server` 仓库执行。Secret 的值不要写进变量、代码或文档。

```bash
# 私密凭据
gh secret set DASHSCOPE_API_KEY --repo nasuf/Companion_server
gh secret set DEEPSEEK_API_KEY --repo nasuf/Companion_server
gh secret set ARK_API_KEY --repo nasuf/Companion_server
gh secret set MINIMAX_API_KEY --repo nasuf/Companion_server

# 公共 API 地址
gh variable set DASHSCOPE_BASE_URL \
  --body 'https://dashscope.aliyuncs.com/compatible-mode/v1' \
  --repo nasuf/Companion_server
# 可选：仅当百炼 workspace 控制台提供独立兼容地址时设置
gh variable set DASHSCOPE_CHARACTER_BASE_URL \
  --body 'https://WORKSPACE_ID.cn-beijing.maas.aliyuncs.com/compatible-mode/v1' \
  --repo nasuf/Companion_server
gh variable set DEEPSEEK_BASE_URL \
  --body 'https://api.deepseek.com' \
  --repo nasuf/Companion_server
gh variable set ARK_BASE_URL \
  --body 'https://ark.cn-beijing.volces.com/api/v3' \
  --repo nasuf/Companion_server
gh variable set MINIMAX_BASE_URL \
  --body 'https://api.minimaxi.com/v1' \
  --repo nasuf/Companion_server
```

`REMOTE_CHAT_PROVIDER`、`REMOTE_SMALL_PROVIDER`、`REMOTE_CHAT_MODEL` 和
`REMOTE_SMALL_MODEL` 也已进入部署 workflow，但生产环境通常应在 Web 后台保存，
由数据库热更新。GitHub Variables 仅作为数据库没有配置时的环境 fallback。

## 2. 通义千问 Character（阿里云百炼）

1. 在阿里云百炼开通 Model Studio，创建或选择 workspace，并创建 API Key。
2. 在角色扮演模型页面确认账号可以调用目标模型。
3. 将 API Key 保存为 GitHub Secret `DASHSCOPE_API_KEY`。
4. 当前账号可直接复用 `DASHSCOPE_BASE_URL`。只有 workspace 控制台明确提供独立
   OpenAI-compatible 地址时，才保存可选变量 `DASHSCOPE_CHARACTER_BASE_URL`；
   普通千问始终使用 `DASHSCOPE_BASE_URL`。
5. 部署后，模型库会自动出现：
   - `qwen-plus-character`
   - `qwen-flash-character`
   - `qwen-flash-character-2026-02-26`
6. 在「模型配置」中把“主回复平台”选为“阿里云百炼 / DashScope”，选择目标
   Character 模型；辅助任务可以继续使用 `qwen3.5-flash`。

官方文档：<https://help.aliyun.com/zh/model-studio/role-play>

## 3. 豆包 Character / 猫箱（火山方舟）

“猫箱”是产品；服务器接入火山方舟公开的豆包角色模型或账号自建推理接入点。

1. 在火山方舟控制台开通模型服务并创建 API Key。
2. 开通 `Doubao-Seed-Character`。模型库会自动预置当前版本
   `doubao-seed-character-260628`；如果账号使用自建推理接入点，也可另外注册
   控制台展示的 `ep-...` ID。
3. 将 API Key 保存为 GitHub Secret `ARK_API_KEY`。
4. 部署后在「模型库」确认模型已出现。预置元数据为 131072 上下文，输入
   ¥0.8/百万 tokens、输出 ¥2/百万 tokens、缓存输入 ¥0.16/百万 tokens；缓存
   存储 ¥0.017/百万 tokens/小时会写在备注中，不进入 token 调用成本估算。
5. 在「模型配置」把主回复平台和模型切到该行。若调用返回 `ModelNotOpen`，需先
   回到火山方舟“开通管理”完成模型服务开通。

官方产品与 API 文档：<https://www.volcengine.com/product/doubao>、
<https://api.volcengine.com/api-docs/view/overview?serviceCode=ark>

## 4. MiniMax M2-her（星野技术路线）

“星野”是产品；MiniMax 当前公开的角色扮演 API 模型是 `M2-her`。

1. 在 MiniMax 开放平台注册、完成认证并创建 API Key。
2. 将 API Key 保存为 GitHub Secret `MINIMAX_API_KEY`。
3. 保持 `MINIMAX_BASE_URL=https://api.minimaxi.com/v1`。
4. 部署后模型库会自动出现并启用 `M2-her`；在「模型配置」选择 MiniMax 与
   `M2-her`。模型库预置 65536 上下文，输入 ¥2.1/百万 tokens、输出
   ¥8.4/百万 tokens；官方未提供提示缓存价格。该模型按官方建议使用
   temperature `1.0`，最大输出 2048 tokens。

官方文档：<https://platform.minimaxi.com/docs/guides/text-chat>、
<https://platform.minimaxi.com/docs/guides/pricing-paygo>

## 5. 上线与验收顺序

1. 先配置所需 Secret/Variable，再触发或等待 Server workflow 部署。
2. 打开 Web 后台模型配置，确认对应供应商显示“凭据已配置”。
3. 在模型库确认/新增精确 model id；禁用的模型不会进入选择框。
4. 先对单个 agent 设置 override 做 A/B 测试，确认语气、延迟、错误率和成本。
5. 验证稳定后再保存为全局主回复模型。辅助任务建议先保留现有低成本小模型。

切换是热更新：保存后下一次 LLM 调用使用新配置；已经在流式生成中的回复仍由
旧模型完成。远程失败时现有本地 Ollama fallback 机制保持不变。
