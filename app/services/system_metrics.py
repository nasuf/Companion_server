"""Host-level resource metrics for the admin 资源监控 dashboard.

Dependency-free by design: uses only the Python standard library (``shutil``
for disk usage plus Linux ``/proc`` parsing). This keeps the deploy image and
``uv.lock`` untouched. Every collector is defensive — on a non-Linux host
(local macOS dev) or when a file is missing, it returns ``None`` for that
dimension instead of raising, so the endpoint never 500s on a metric gap.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time

logger = logging.getLogger(__name__)

# Sampling window for rate-based metrics (CPU %, network throughput). Kept short
# so the endpoint stays responsive; the event loop is freed via asyncio.sleep.
_SAMPLE_INTERVAL_S = 0.25

# Disk mounts to report. The server container mounts the CVM data disk under
# /data/*, so probing a data path reflects the disk where Postgres/Redis/media
# actually live (that is the disk that fills up as agents are cloned).
_SYSTEM_MOUNT = "/"


def _data_mount() -> str:
    """Path that resolves onto the data disk inside the server container."""
    return os.getenv("CHAT_MEDIA_DIR") or "/data/chat_media"


def _read_proc(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


def _collect_disks() -> list[dict]:
    """Disk usage per distinct filesystem (system disk + data disk)."""
    disks: list[dict] = []
    seen_devices: set[int] = set()
    candidates = [(_SYSTEM_MOUNT, "系统盘"), (_data_mount(), "数据盘")]
    for mount, label in candidates:
        try:
            device = os.stat(mount).st_dev
        except OSError:
            continue
        if device in seen_devices:
            continue
        try:
            usage = shutil.disk_usage(mount)
        except OSError:
            continue
        seen_devices.add(device)
        percent = round(usage.used / usage.total * 100, 1) if usage.total else 0.0
        disks.append({
            "mount": mount,
            "label": label,
            "total": usage.total,
            "used": usage.used,
            "free": usage.free,
            "percent": percent,
        })
    return disks


def _parse_meminfo() -> dict | None:
    raw = _read_proc("/proc/meminfo")
    if not raw:
        return None
    values: dict[str, int] = {}
    for line in raw.splitlines():
        parts = line.split(":")
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        num = parts[1].strip().split()
        if num and num[0].isdigit():
            # /proc/meminfo is in kB.
            values[key] = int(num[0]) * 1024
    total = values.get("MemTotal")
    if not total:
        return None
    available = values.get("MemAvailable")
    if available is None:
        available = values.get("MemFree", 0) + values.get("Buffers", 0) + values.get("Cached", 0)
    used = max(total - available, 0)
    swap_total = values.get("SwapTotal", 0)
    swap_free = values.get("SwapFree", 0)
    swap_used = max(swap_total - swap_free, 0)
    return {
        "total": total,
        "used": used,
        "available": available,
        "percent": round(used / total * 100, 1),
        "swap_total": swap_total,
        "swap_used": swap_used,
        "swap_percent": round(swap_used / swap_total * 100, 1) if swap_total else 0.0,
    }


def _read_cpu_times() -> tuple[int, int] | None:
    """Return (idle, total) jiffies from the aggregate cpu line in /proc/stat."""
    raw = _read_proc("/proc/stat")
    if not raw:
        return None
    for line in raw.splitlines():
        if line.startswith("cpu "):
            fields = [int(x) for x in line.split()[1:] if x.isdigit()]
            if len(fields) < 5:
                return None
            idle = fields[3] + fields[4]  # idle + iowait
            total = sum(fields)
            return idle, total
    return None


def _read_net_bytes() -> dict | None:
    """Aggregate rx/tx bytes and packets across non-loopback interfaces."""
    raw = _read_proc("/proc/net/dev")
    if not raw:
        return None
    rx_bytes = tx_bytes = rx_pkts = tx_pkts = 0
    for line in raw.splitlines():
        if ":" not in line:
            continue
        iface, _, rest = line.partition(":")
        iface = iface.strip()
        if iface == "lo":
            continue
        cols = rest.split()
        if len(cols) < 16:
            continue
        rx_bytes += int(cols[0])
        rx_pkts += int(cols[1])
        tx_bytes += int(cols[8])
        tx_pkts += int(cols[9])
    return {
        "bytes_recv": rx_bytes,
        "bytes_sent": tx_bytes,
        "packets_recv": rx_pkts,
        "packets_sent": tx_pkts,
    }


def _uptime_seconds() -> float | None:
    raw = _read_proc("/proc/uptime")
    if not raw:
        return None
    try:
        return round(float(raw.split()[0]), 1)
    except (ValueError, IndexError):
        return None


async def collect_host_metrics() -> dict:
    """Collect CPU / memory / disk / network metrics for the host.

    CPU % and network throughput are rate metrics, so we sample twice around a
    short ``asyncio.sleep`` (non-blocking). All fields are best-effort.
    """
    cpu_first = _read_cpu_times()
    net_first = _read_net_bytes()
    sampled_at = time.monotonic()

    if cpu_first is not None or net_first is not None:
        await asyncio.sleep(_SAMPLE_INTERVAL_S)

    elapsed = max(time.monotonic() - sampled_at, 1e-6)

    cpu_percent: float | None = None
    cpu_second = _read_cpu_times()
    if cpu_first and cpu_second:
        idle_delta = cpu_second[0] - cpu_first[0]
        total_delta = cpu_second[1] - cpu_first[1]
        if total_delta > 0:
            cpu_percent = round((1 - idle_delta / total_delta) * 100, 1)

    net: dict | None = None
    net_second = _read_net_bytes()
    if net_first and net_second:
        net = dict(net_second)
        net["recv_rate_bps"] = max(
            round((net_second["bytes_recv"] - net_first["bytes_recv"]) / elapsed), 0
        )
        net["sent_rate_bps"] = max(
            round((net_second["bytes_sent"] - net_first["bytes_sent"]) / elapsed), 0
        )

    load_avg: list[float] | None = None
    try:
        load_avg = [round(x, 2) for x in os.getloadavg()]
    except (OSError, AttributeError):
        load_avg = None

    cpu_count = os.cpu_count()

    return {
        "system": {
            "cpu_count": cpu_count,
            "cpu_percent": cpu_percent,
            "load_avg": load_avg,
            "load_percent": (
                round(load_avg[0] / cpu_count * 100, 1)
                if load_avg and cpu_count
                else None
            ),
            "uptime_seconds": _uptime_seconds(),
        },
        "memory": _parse_meminfo(),
        "disks": _collect_disks(),
        "network": net,
    }
