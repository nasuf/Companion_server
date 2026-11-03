"""Offline IPv4 region/ISP lookup over the bundled ip2region xdb (v3 structure).

Data: app/data/ip2region/ip2region_v4.xdb.gz, upstream lionsoul2014/ip2region
(Apache-2.0 or MIT, see LICENSE.md next to it). No network call is made; the
decompressed 11 MB buffer is loaded once per process on first use.

Only the lookup half of the upstream binding is needed: a 256x256 vector index
narrows the search to one segment block, then a binary search over 14-byte
records (start ip, end ip, data length, data pointer; little endian).
"""
from __future__ import annotations

import gzip
import ipaddress
import struct
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "ip2region" / "ip2region_v4.xdb.gz"
_HEADER = 256
_VECTOR_COLS = 256
_VECTOR_SIZE = 8
_SEGMENT = 14


@dataclass(frozen=True, slots=True)
class IpRegion:
    country: str | None
    province: str | None
    city: str | None
    isp: str | None


_EMPTY = IpRegion(None, None, None, None)


@lru_cache(maxsize=1)
def _buffer() -> bytes:
    data = gzip.decompress(DATA_PATH.read_bytes())
    version, ip_version = struct.unpack_from("<H", data, 0)[0], struct.unpack_from("<H", data, 16)[0]
    if version != 3 or ip_version != 4:
        raise ValueError("Unsupported ip2region data")
    return data


def _field(value: str) -> str | None:
    value = value.strip()
    return None if value in {"", "0", "Reserved"} else value


def lookup(ip: str) -> IpRegion:
    """Region of a public IPv4 address. IPv6, private and unknown -> all None."""
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return _EMPTY
    if address.version != 4 or not address.is_global:
        return _EMPTY
    data = _buffer()
    value = int(address)
    b0, b1 = value >> 24, (value >> 16) & 0xFF
    start, end = struct.unpack_from(
        "<II", data, _HEADER + (b0 * _VECTOR_COLS + b1) * _VECTOR_SIZE)
    if not start or not end:
        return _EMPTY
    low, high = 0, (end - start) // _SEGMENT
    while low <= high:
        mid = (low + high) >> 1
        first, last, length, pointer = struct.unpack_from("<IIHI", data, start + mid * _SEGMENT)
        if value < first:
            high = mid - 1
        elif value > last:
            low = mid + 1
        else:
            parts = data[pointer:pointer + length].decode("utf-8").split("|")
            if len(parts) < 4:
                return _EMPTY
            return IpRegion(*(_field(p) for p in parts[:4]))
    return _EMPTY


# Hosting and cloud operators as labeled in the bundled data (checked against
# its full label list, 2026-11). A real phone on a carrier network never shows
# these; a relay or script running on a server does.
_DATACENTER_MARKERS = (
    "阿里", "腾讯", "华为", "百度", "金山云", "京东", "字节跳动", "网宿", "世纪互联", "青云",
    "美团云", "网易云", "蓝云", "飞云", "UCloud", "Amazon", "Microsoft", "Akamai",
    "Cloudflare", "Fastly", "Linode", "DigitalOcean", "Vultr", "Choopa", "OVH", "Hetzner",
    "Oracle", "IBM Cloud", "Alibaba", "Tencent", "HUAWEI CLOUD", "Hosting", "Hostinger",
    "IDC", "Data Center", "Datacenter", "Data Centre", "Datacamp", "GTHost", "Leaseweb",
)
_CARRIER_MARKERS = (
    "电信", "联通", "移动", "铁通", "广电", "有线", "鹏博士", "长城宽带", "电讯", "宽频",
)


def network_kind(region: IpRegion) -> str:
    """datacenter / campus / carrier / overseas / unknown."""
    isp = region.isp or ""
    if any(marker in isp for marker in _DATACENTER_MARKERS) or (
        isp.startswith("Google") and "Fiber" not in isp
    ):
        return "datacenter"
    if "教育网" in isp or "科技网" in isp:
        return "campus"
    if region.country and region.country != "中国":
        return "overseas"
    if any(marker in isp for marker in _CARRIER_MARKERS):
        return "carrier"
    return "unknown"
