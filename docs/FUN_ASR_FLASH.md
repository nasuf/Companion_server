# Fun-ASR-Flash 聊天语音输入

聊天页通过 Flutter `record` 插件录制 16 kHz、单声道、32 kbps AAC/M4A，随后把音频以 Base64 JSON 上传到已鉴权的 `POST /chat/transcribe`。该聊天接口只接收 M4A，并从容器 `mvhd` 读取真实时长，不信任客户端上报值；会话归属、格式、真实时长和文件大小均通过后才调用 DashScope Fun-ASR-Flash。选择“发送语音”且识别成功后，原始 M4A 写入 `CHAT_MEDIA_DIR`，数据库 `chat_message_attachments` 写入音频链接、时长、转写文本、模型和 request ID。

Flutter 随后通过现有聊天 WebSocket 发送转写文字；只有“发送语音”会额外携带音频附件 ID。后端把转写文字保存到 `messages.content`，并在有附件时把它绑定到该消息，因此：

- AI 意图、记忆和回复模型继续读取文字，不处理二进制音频；
- 未来聊天搜索直接检索 `messages.content`；
- “发送语音”会显示可播放的语音气泡，并保存原音频；
- “转文字发送”只发送并保存文字，原音频仅存在于当次 ASR 请求内，不落盘、不建附件；
- 取消录音不会上传，也不会产生数据库或磁盘文件。

客户端上传成功或失败后都会删除本地临时文件。ASR 失败时服务端不创建附件；ASR 成功但数据库写入失败时会立即删除刚写入的磁盘文件。后端日志只记录时长、模型和 DashScope request ID，不记录音频或转写正文。

## 必需配置

1. 在阿里云百炼开通模型服务，并在**华北 2（北京）**地域创建 API Key。
2. 在 `Companion_server/.env` 中设置：

```dotenv
DASHSCOPE_API_KEY=sk-替换为北京地域的百炼APIKey
```

如果服务端已经用同一个北京地域的 `DASHSCOPE_API_KEY` 调用通义模型，无需再申请第二把 Key。

还需要保证生产容器继续挂载持久数据盘：

```yaml
/mnt/datadisk0/companion/chat_media:/data/chat_media
```

并设置 `CHAT_MEDIA_DIR=/data/chat_media`。Flutter 不需要任何百炼密钥。

上线前必须执行数据库迁移 `20260717120000_chat_audio_attachments`，再重启后端。

## 可选参数

以下均已有生产可用默认值，需要时可在 `.env` 覆盖：

```dotenv
DASHSCOPE_ASR_MODEL=fun-asr-flash-2026-06-15
DASHSCOPE_ASR_ENDPOINT=https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
DASHSCOPE_ASR_TIMEOUT_S=30
CHAT_VOICE_MIN_SECONDS=0.5
CHAT_VOICE_MAX_SECONDS=60
CHAT_VOICE_MAX_REQUESTS_PER_MINUTE=20
CHAT_VOICE_MAX_BYTES=2097152
```

若使用百炼子业务空间专属域名，把端点改为：

```dotenv
DASHSCOPE_ASR_ENDPOINT=https://你的WorkspaceId.cn-beijing.maas.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation
```

`CHAT_VOICE_MIN_SECONDS`、`CHAT_VOICE_MAX_SECONDS` 和 `CHAT_VOICE_MAX_BYTES` 是服务端最终安全上限，服务端会从 M4A 容器核验真实时长，低于 0.5 秒的无效录音不会请求 ASR；`CHAT_VOICE_MAX_REQUESTS_PER_MINUTE` 是每用户的 ASR 防刷上限。Flutter 当前也分别限制为 60 秒和 2 MiB；若要放宽，应同步修改客户端常量，但不能超过 Fun-ASR-Flash 的 5 分钟限制，且 Base64 编码后的数据应小于 10 MB。

## 识别质量

后端会把当前会话最近 10 条用户/AI 消息作为上下文，放在音频消息之前。每条最多 200 字，因此同一轮用户与 AI 文本最多 400 字，符合 Fun-ASR-Flash 上下文限制。上下文能帮助识别人名、地名和聊天中刚出现的专有名词。

当前录音参数：

- AAC-LC / M4A
- 16 kHz
- 单声道
- 32 kbps
- 客户端降噪开启

## 磁盘容量

32 kbps 约等于每秒 4 KB：

- 10 秒约 40 KB；
- 30 秒约 120 KB；
- 60 秒约 240 KB；
- 1,000 个日活用户、每人每天 10 分钟，约 2.4 GB/天、72 GB/月（不含备份副本）。

单条语音通常不比聊天图片大：当前图片上传会压到质量 82、最长边 1600，常见仍是数百 KB 到数 MB；60 秒语音约 240 KB。语音的主要风险是用户每天累计分钟数稳定增长，因此应监控 `chat_media` 目录总量，并把它纳入与数据库同周期的备份和恢复演练。

## 上线检查

- iOS 的 `NSMicrophoneUsageDescription` 已包含聊天语音输入；发布新包后用户首次使用会看到系统授权框。
- Android 已声明 `android.permission.RECORD_AUDIO`，无需新增 Manifest 配置。
- 后端容器必须能访问 `dashscope.aliyuncs.com:443`。
- 数据盘 `chat_media` 目录必须可写，并与数据库备份保持同一恢复点；只恢复数据库或只恢复文件都会产生失效链接或孤儿文件。
- 生产日志出现 `[speech-to-text] chat transcription` 且带 `request_id`，说明整条识别链路成功。
- `503` 通常表示 Key/模型/端点未配置或 Key 地域不匹配；`429` 是百炼限流；`504` 是识别超时；`422` 表示没有识别到清晰语音或超过时长限制。
