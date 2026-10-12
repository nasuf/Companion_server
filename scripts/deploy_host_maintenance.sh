#!/usr/bin/env bash
# Sourced after server health succeeds; keep existing cleanup and cron commands.
: "${DOCKER:?DOCKER must be set by the deployment caller}"

# Dense releases must not retain an unbounded number of full image copies.
# Keep at least three fully verified rollback copies and the latest pointer.
python3 scripts/release_archive.py prune --root /app/companion-release-archives/server \
  --max-archives 5 --max-bytes 4294967296 || logger -p user.err -t companion-archive 'post-deploy rotation failed'
bash scripts/install_release_archive_maintenance.sh "$(id -un)"

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

echo "==> Installing the canonical database backup timer"
bash scripts/install_postgres_backup_timer.sh
