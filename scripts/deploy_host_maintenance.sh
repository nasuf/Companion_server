#!/usr/bin/env bash
# Sourced after server health succeeds; keep existing cleanup and cron commands.
: "${DOCKER:?DOCKER must be set by the deployment caller}"

# Archive retention is separate from Docker prune: keep 14 days and at least
# three verified rollback artifacts, including the latest pointer. Failure does
# not remove any artifacts or undo an already healthy deployment.
python3 scripts/release_archive.py prune --root /app/companion-release-archives/server || true

echo "==> Cleaning up old docker images (>72h old, dangling, untagged)"
# 防止历史 deploy 镜像层堆积撑爆磁盘 (Redis bgsave 一旦失败就锁所有
# 写命令, 整个后端起不来). until=72h 安全保留近 3 天回滚目标; -a
# 包含未被任何容器引用的具名 image, -f 跳过交互确认.
# builder 缓存用 keep-storage 硬上限而非时间过滤: 密集部署期 (一天
# 多次 push) 72h 窗口内的缓存也能堆到 15GB+ (2026-07-07 实测把系统
# 盘吃到 74%), 按总量兜底才是有效边界; 2GB 足够保住最近一次构建的
# 层缓存让增量构建仍然快.
$DOCKER image prune -af --filter "until=72h" || true
$DOCKER builder prune -af --keep-storage=2GB || true

echo "==> Installing weekly disk cleanup cron (idempotent)"
# 部署只在 push 时发生, 间隔可能数周; journal/apt/构建残渣的增长与
# 部署无关, 所以再挂一个每周日 04:30 的 host cron 兜底. 策略全部
# 有界: 不碰 volumes/数据盘, 镜像保留 72h 回滚窗口.
sudo tee /usr/local/bin/companion-disk-cleanup.sh >/dev/null <<'CLEANUP'
#!/bin/sh
# Companion stack weekly disk cleanup (installed by deploy.yml; safe to re-run).
# Bounded policies only — never touches volumes/data, keeps 72h rollback images.
docker image prune -af --filter "until=72h" || true
docker builder prune -af --keep-storage=2GB || true
journalctl --vacuum-size=200M || true
apt-get clean || true
df -h / | tail -1 | logger -t companion-cleanup || true
CLEANUP
sudo chmod 755 /usr/local/bin/companion-disk-cleanup.sh
sudo tee /etc/cron.d/companion-disk-cleanup >/dev/null <<'CRONTAB'
30 4 * * 0 root /usr/local/bin/companion-disk-cleanup.sh >/dev/null 2>&1
CRONTAB
sudo chmod 644 /etc/cron.d/companion-disk-cleanup

echo "==> Installing nightly database backup cron (idempotent)"
# 2026-07-28: 一条过宽的 DELETE 清掉 1691 行变更日志后才发现整台机器
# 没有任何备份 —— 无定时 dump, archive_mode=off, 无 WAL 归档。库只有
# 294MB, 保留 14 天也就 1GB 上下, 没有不做的理由。
# 脚本源文件在 scripts/companion-db-backup.sh, 改那边后这里要同步。
sudo tee /usr/local/bin/companion-db-backup.sh >/dev/null <<'BACKUP'
#!/bin/sh
# Companion 数据库每日备份 (由 deploy.yml 幂等安装; 可随时手工重跑)。
#
# 装这个的直接原因: 2026-07-28 一条过宽的 DELETE 清掉了 1691 行 access 变更日志,
# 排查时才发现整台机器没有任何备份 —— 没有定时 dump, archive_mode=off, 也没有
# WAL 归档。当天丢的是可再生的日志; 下次若是 memories_* 就没有退路。
#
# 用 custom 格式 (-Fc) 而不是纯 SQL: 它支持只恢复单张表。事故形态通常是"某一张
# 表被误改", 全库回滚会把其它表这段时间的正常写入一起抹掉, 那是第二次事故。
#
# 恢复单表:
#   pg_restore -U companion -d companion -t memory_changelogs --data-only \
#     --disable-triggers /var/backups/companion/companion-YYYYmmdd-HHMM.dump
# 恢复全库 (先建空库):
#   pg_restore -U companion -d companion --clean --if-exists <dump>

set -eu

BACKUP_DIR=/var/backups/companion
CONTAINER=companion-postgres
DB_USER=companion
DB_NAME=companion
KEEP_DAYS=14

mkdir -p "$BACKUP_DIR"
STAMP="$(date +%Y%m%d-%H%M)"
TARGET="$BACKUP_DIR/companion-$STAMP.dump"

fail() {
    # 备份失败必须响亮 —— 一个每晚静默失败的备份比没有备份更危险, 因为你以为有。
    logger -t companion-backup -p user.err "BACKUP FAILED: $1"
    echo "companion-backup FAILED: $1" >&2
    exit 1
}

docker exec "$CONTAINER" pg_dump -U "$DB_USER" -d "$DB_NAME" -Fc \
    > "$TARGET".partial 2>/dev/null || fail "pg_dump 退出码非零"

# 先落 .partial 再改名: 中途被打断时不会留下一个看起来完好的残缺备份。
mv "$TARGET".partial "$TARGET"

# 没验证过能读的备份不算备份。pg_restore --list 会解析归档头和目录,
# 文件截断或损坏在这里就会暴露, 不用等到真出事那天。
TABLES="$(docker exec -i "$CONTAINER" pg_restore --list < "$TARGET" 2>/dev/null | grep -c 'TABLE DATA' || true)"
[ "${TABLES:-0}" -ge 20 ] || fail "校验失败: 归档里只有 ${TABLES:-0} 张表的数据"

SIZE="$(du -h "$TARGET" | cut -f1)"
logger -t companion-backup "ok $STAMP size=$SIZE tables=$TABLES"

# 轮转。先确认新备份存在再删旧的, 避免"删完才发现今天没备成"。
[ -s "$TARGET" ] || fail "备份文件为空"
find "$BACKUP_DIR" -name 'companion-*.dump' -mtime +"$KEEP_DAYS" -delete || true
find "$BACKUP_DIR" -name '*.partial' -mtime +1 -delete || true

df -h / | tail -1 | logger -t companion-backup || true
BACKUP
sudo chmod 755 /usr/local/bin/companion-db-backup.sh
sudo tee /etc/cron.d/companion-db-backup >/dev/null <<'BACKUPCRON'
# 03:00 —— 避开 02:30 的 L2 动态分级和 03:30 的作息生成。
0 3 * * * root /usr/local/bin/companion-db-backup.sh >/dev/null 2>&1
BACKUPCRON
sudo chmod 644 /etc/cron.d/companion-db-backup
