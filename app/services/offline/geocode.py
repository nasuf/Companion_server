"""线下活动地理编码与到达距离计算。

- geocode_address(): 地址 -> (lat, lng)，用高德 geocode/geo（v3）。未配置 key 或失败
  时返回 None（优雅降级，不阻断推荐生成；到达校验侧按
  settings.offline_arrival_require_geocode 决定拦截或放行）。
- haversine_m(): 两经纬度直线距离（米），供 arrive() ≤200m 校验。
- make_place_key(): 由地点名/地址/城市生成稳定去重键，供「同地点复用进行中活动」。
"""

from __future__ import annotations

import hashlib
import logging
import math
import re

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

_EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """两点直线距离（米）。"""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def make_place_key(
    location_name: str | None,
    address: str | None,
    city: str | None = None,
) -> str | None:
    """稳定去重键：归一化 (地点名 + 地址 + 城市) 后取短 hash。

    用于 spec §4.4「同地点已有进行中活动时再次接受，直接打开既有打卡页」。
    """
    parts = [p for p in (location_name, address, city) if p]
    if not parts:
        return None
    raw = "|".join(re.sub(r"\s+", "", str(p)).lower() for p in parts)
    if not raw:
        return None
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


async def geocode_address(
    address: str | None,
    city: str | None = None,
) -> tuple[float, float] | None:
    """地址 -> (lat, lng)。未配置 key / 无地址 / 失败 -> None。"""
    if not address:
        return None
    key = (settings.amap_geocode_key or "").strip()
    if not key:
        logger.info("[offline-geocode] amap_geocode_key 未配置，跳过地理编码")
        return None
    params = {"key": key, "address": address}
    if city:
        params["city"] = city
    try:
        async with httpx.AsyncClient(
            timeout=settings.offline_geocode_timeout_s, trust_env=False
        ) as client:
            resp = await client.get(settings.amap_geocode_endpoint, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:  # 网络/超时/解析失败一律降级
        logger.warning("[offline-geocode] 地理编码失败 address=%r err=%s", address, exc)
        return None

    if str(data.get("status")) != "1":
        logger.warning(
            "[offline-geocode] amap 返回非成功 status=%s info=%s address=%r",
            data.get("status"), data.get("info"), address,
        )
        return None
    geocodes = data.get("geocodes") or []
    if not geocodes:
        return None
    if geocodes[0].get("level") in {"国家", "省", "市", "区县", "开发区", "乡镇", "村庄", "未知"}:
        return None  # A district centroid cannot verify arrival at a particular place.
    location = str(geocodes[0].get("location") or "")  # 高德格式："lng,lat"
    try:
        lng_str, lat_str = location.split(",", 1)
        lat, lng = float(lat_str), float(lng_str)
    except (ValueError, AttributeError):
        logger.warning("[offline-geocode] 无法解析 location=%r", location)
        return None
    if not (-90 <= lat <= 90 and -180 <= lng <= 180):
        return None
    return lat, lng


def wgs84_to_gcj02(lat: float, lng: float) -> tuple[float, float]:
    """Align system GPS with the coordinate system returned by Amap.

    Outside mainland China's transform bounds the coordinates are unchanged.
    This is a local calculation; it performs no paid API request.
    """
    if not (72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271):
        return lat, lng
    x, y = lng - 105, lat - 35
    dlat = -100 + 2*x + 3*y + .2*y*y + .1*x*y + .2*math.sqrt(abs(x))
    dlat += (20*math.sin(6*x*math.pi)+20*math.sin(2*x*math.pi))*2/3
    dlat += (20*math.sin(y*math.pi)+40*math.sin(y/3*math.pi))*2/3
    dlat += (160*math.sin(y/12*math.pi)+320*math.sin(y*math.pi/30))*2/3
    dlng = 300+x+2*y+.1*x*x+.1*x*y+.1*math.sqrt(abs(x))
    dlng += (20*math.sin(6*x*math.pi)+20*math.sin(2*x*math.pi))*2/3
    dlng += (20*math.sin(x*math.pi)+40*math.sin(x/3*math.pi))*2/3
    dlng += (150*math.sin(x/12*math.pi)+300*math.sin(x/30*math.pi))*2/3
    rad = lat / 180 * math.pi
    magic = 1 - .00669342162296594323 * math.sin(rad)**2
    dlat = dlat*180 / ((6378245*(1-.00669342162296594323))/(magic*math.sqrt(magic))*math.pi)
    dlng = dlng*180 / (6378245/math.sqrt(magic)*math.cos(rad)*math.pi)
    return lat+dlat, lng+dlng
