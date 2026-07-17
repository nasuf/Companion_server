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
gh secret set QIANFAN_API_KEY --repo nasuf/Companion_server
gh secret set ARK_API_KEY --repo nasuf/Companion_server
gh secret set MINIMAX_API_KEY --repo nasuf/Companion_server

# 公共 API 地址
gh variable set DASHSCOPE_CHARACTER_BASE_URL \
  --body 'https://WORKSPACE_ID.cn-beijing.maas.aliyuncs.com/compatible-mode/v1' \
  --repo nasuf/Companion_server
gh variable set QIANFAN_BASE_URL \
  --body 'https://qianfan.baidubce.com/v2' \
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
4. 将 workspace 专属 OpenAI-compatible 地址保存为 GitHub Variable
   `DASHSCOPE_CHARACTER_BASE_URL`。普通千问仍使用 `DASHSCOPE_BASE_URL`。
5. 部署后，模型库会自动出现：
   - `qwen-plus-character`
   - `qwen-flash-character`
   - `qwen-flash-character-2026-02-26`
6. 在「模型配置」中把“主回复平台”选为“阿里云百炼 / DashScope”，选择目标
   Character 模型；辅助任务可以继续使用 `qwen3.5-flash`。

官方文档：<https://help.aliyun.com/zh/model-studio/role-play>

## 3. ERNIE Character（百度千帆）

百度已经退役预置的 `ERNIE-Character-8K`（2026-06-09）以及更早的 Fiction
版本，所以新账号无法按截图名称重新开通。迁移会把该模型作为“已禁用”历史行
显示，防止管理员误选。

如果千帆账号内仍有可用的自定义角色模型/服务接入点：

1. 在千帆 ModelBuilder 创建 API Key，并保存为 GitHub Secret
   `QIANFAN_API_KEY`。
2. 保持 `QIANFAN_BASE_URL=https://qianfan.baidubce.com/v2`，或者填账号文档指定
   的 OpenAI-compatible 地址。
3. 在「模型库」新增一行：Provider 选“百度千帆”，Identifier 填 API 实际接受
   的模型或接入点 ID，启用后保存。
4. 在「模型配置」选择该平台与模型。

官方文档：<https://cloud.baidu.com/doc/qianfan-docs/s/qm8qxemze>、
<https://cloud.baidu.com/doc/qianfan/s/zmh4stou3>

## 4. 豆包 Character / 猫箱（火山方舟）

“猫箱”是产品，不是稳定的公共 API model id；服务器接入的是火山方舟中账号
实际可用的豆包角色模型或推理接入点。

1. 在火山方舟控制台开通模型服务并创建 API Key。
2. 开通 `Doubao-Seed-Character`，按控制台创建推理接入点；复制控制台展示的
   model id 或 endpoint id。不同账号/区域的 ID 可能不同，代码不预置猜测值。
3. 将 API Key 保存为 GitHub Secret `ARK_API_KEY`。
4. 在「模型库」新增一行：Provider 选“火山方舟 / 豆包”，Identifier 原样粘贴
   上一步 ID，填写控制台对应上下文长度和价格，启用后保存。
5. 在「模型配置」把主回复平台和模型切到该行。

官方产品与 API 文档：<https://www.volcengine.com/product/doubao>、
<https://api.volcengine.com/api-docs/view/overview?serviceCode=ark>

## 5. MiniMax M2-her（星野技术路线）

“星野”是产品；MiniMax 当前公开的角色扮演 API 模型是 `M2-her`。

1. 在 MiniMax 开放平台注册、完成认证并创建 API Key。
2. 将 API Key 保存为 GitHub Secret `MINIMAX_API_KEY`。
3. 保持 `MINIMAX_BASE_URL=https://api.minimaxi.com/v1`。
4. 部署后模型库会自动出现并启用 `M2-her`；在「模型配置」选择 MiniMax 与
   `M2-her`。该模型按官方建议使用 temperature `1.0`。

官方文档：<https://platform.minimaxi.com/docs/guides/text-chat>、
<https://platform.minimaxi.com/docs/api-reference/api-overview>

## 6. 上线与验收顺序

1. 先配置所需 Secret/Variable，再触发或等待 Server workflow 部署。
2. 打开 Web 后台模型配置，确认对应供应商显示“凭据已配置”。
3. 在模型库确认/新增精确 model id；禁用的模型不会进入选择框。
4. 先对单个 agent 设置 override 做 A/B 测试，确认语气、延迟、错误率和成本。
5. 验证稳定后再保存为全局主回复模型。辅助任务建议先保留现有低成本小模型。

切换是热更新：保存后下一次 LLM 调用使用新配置；已经在流式生成中的回复仍由
旧模型完成。远程失败时现有本地 Ollama fallback 机制保持不变。
