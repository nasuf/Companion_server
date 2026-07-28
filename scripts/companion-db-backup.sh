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
