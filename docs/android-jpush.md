# Android 极光推送接入

本次使用官方 Android SDK 6.2.1，复用 Flutter 原有 MethodChannel；iOS 继续使用 APNs，无新增 iOS SDK。无需数据库结构迁移。

## 配置

在极光控制台分别创建应用，包名必须与安装包一致：
- 正式版 `com.bansheng.prod`
- 开发版 `com.bansheng.dev`

构建环境设置 `JPUSH_PROD_APPKEY` / `JPUSH_DEV_APPKEY`，也可放入用户级 `~/.gradle/gradle.properties` 同名属性。AppKey 可随 APK 分发，Master Secret 只能保存在服务器。

服务端单应用配置（已支持现有变量名，默认仅对应正式包）：

```dotenv
JIGUANG_PUSH_APP_KEY=实际正式版AppKey
JIGUANG_PUSH_MASTER_SECRET=实际正式版MasterSecret
JIGUANG_PUSH_PACKAGE_NAME=com.bansheng.prod
```

两项凭据完整时自动启用匹配包名的发送，无需再设置 `JPUSH_ENABLED=true`。默认包名为 `com.bansheng.prod`，不会将这组凭据用于开发包。可显式设置 `JPUSH_ENABLED=false` 紧急关闭全部极光发送。Master Secret 使用敏感值类型，配置对象打印时隐藏其内容。

若以后两个应用均开通，可用多应用配置：

```dotenv
JPUSH_ENABLED=true
JPUSH_APPS={"com.bansheng.prod":{"app_key":"实际正式版AppKey","master_secret":"实际正式版MasterSecret"},"com.bansheng.dev":{"app_key":"实际开发版AppKey","master_secret":"实际开发版MasterSecret"}}
```

同一包名的 `JPUSH_APPS` 条目优先于单应用变量；显式条目无效或 JSON 配置错误时停止使用该配置，不猜测其他密钥。

配置后重启实际执行通知调度的进程。无凭据时不可发送；缺少配置不会将安卓通知记成成功。发布顺序为先兼容服务端、后安装新版 APK。旧 iOS 注册协议兼容。服务端 `.env` 不会自动进入安卓构建，构建机仍需配置上述 `JPUSH_PROD_APPKEY`，且其值必须与服务端 AppKey 相同。

## 链路与验收

### GitHub Actions 生产配置

仓库 `nasuf/Companion_server` 中，`JIGUANG_PUSH_MASTER_SECRET` 放 Actions Secret；`JIGUANG_PUSH_APP_KEY`、`JIGUANG_PUSH_PACKAGE_NAME=com.bansheng.prod` 和 `JPUSH_ENABLED=true` 放 Actions Variables。工作流通过 SSH Action 的环境变量转发，先执行脱敏配置校验，再重建服务器 `.env`。Compose 的 `env_file` 将这些值注入服务进程；仅修改开发机 `.env` 无法更新生产。`JPUSH_ENABLED=false` 是明确关闭开关。多应用 JSON 配置当前仅支持手工环境配置，尚未接入此部署工作流。

Flutter 仓库当前没有 GitHub Actions 构建工作流；APK 在本机构建，使用本机用户级 Gradle 的 `JPUSH_PROD_APPKEY`。以后迁移到 Actions 构建 APK 时，需在 Flutter 仓库单独配置该 Variables；服务端仓库的 Variables 不会自动跨仓库共享。Master Secret 永远不传入 APK 构建。

登录后初始化 SDK、请求 Android 13+ 通知权限、上报 registration ID 和包名。服务端按账号查设备并定向发送；点击通知复用既有会话/成就/胶囊路由；退出登录禁用该用户 token 并停止本地推送。registration ID 原样保存，不经过 APNs 十六进制清洗。

必须在隐私政策/第三方 SDK 清单披露极光及用途，并核对登录前隐私同意流程。客户端启动阶段只建立桥接，登录配置时才 init SDK。

真机验收：授权/拒绝通知、在线前台去重、后台与锁屏、冷启动点击、账号切换/退出、开发与正式包隔离、错误密钥、无配置、网络失败。`sent` 只代表极光接口接受；尚无送达回执，不表示设备展示。

本次只接极光基础通道，未接华为/荣耀/小米/OPPO/vivo 厂商 SDK；不能保证进程回收后离线送达。厂商申请、后台浏览器与生产真机验收另行完成。

已知沿用 outbox 限制：事件级成功后，部分设备失败只保留诊断，不单独重试；失败重试无独立设备幂等账本，进程中断时可能重复通知。不得将该实现描述成 exactly-once 或全设备送达。

## 本地验证记录（2026-10-21）

14 项后端通知测试通过；Flutter 改动的两个 Dart 文件 analyzer 无问题；dev/prod Debug Kotlin 编译通过；两个仓库 diff --check 通过。未配置真实凭据，未部署，未运行生产/后台浏览器或真机送达验收。全项目 Flutter analyzer 仍有 16 项既存提示，不属于本次文件。

### 现有 .env 变量兼容验证

已验证 JIGUANG_PUSH_APP_KEY / JIGUANG_PUSH_MASTER_SECRET 可由实际 Settings 读取，正式包凭据就绪、开发包无匹配凭据；配置检查未向极光发请求，不表示远端凭据有效。23 项通知测试通过（含显式关闭、配置优先级、包名隔离与敏感值隐藏）；prod Debug Kotlin/manifest 构建通过，合并 Manifest 的正式包 AppKey 与后端配置一致。本机用户级 Gradle 已配置正式包 AppKey；未复制 Master Secret 到客户端或仓库。未部署/重启服务，未安装 APK 或运行真机验收。

## 发布前复核

两轮 review 已检查平台路由、包名/凭据隔离、失败诊断、旧 iOS 协议与提交范围。修复安卓初始化返回值类型，并将客户端配置/退出串行化，等待在途设备注册结束后再禁用，防止退出后被晚到的注册重新启用；保留 APNs 格式化 token 的注销兼容性。73 项后端相关回归通过；新增 3 项 Flutter 推送测试通过，改动文件 analyzer 无问题。通知改动行覆盖 66/69，分支 26/26（局部测量，不代替 CI 全量门槛）。正式版 APK 升级为 0.6.1+9。

Flutter 全量本地测试 720 通过、3 跳过、2 失败；两项失败为 capsule/checkin 页面测试，均在未修改的 4a7e9e4 基线复现。生产部署仍须以当前提交的 GitHub CI、部署结果和生产检查为准；没有连接安卓设备，不能宣称真机送达已通过。
