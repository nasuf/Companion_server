"""ENV01.06: production services publish ports to loopback only.

The database used to listen on 0.0.0.0:5432 with no host firewall; containers reach
it as postgres:5432 on the compose network, and operators go through an SSH tunnel.
"""

from pathlib import Path

import yaml

_COMPOSE = Path(__file__).resolve().parents[1] / "docker-compose.deploy.yml"


def test_every_published_port_binds_loopback():
    services = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))["services"]
    published = {
        (name, str(port))
        for name, spec in services.items()
        for port in spec.get("ports", [])
    }
    assert ("postgres", "127.0.0.1:5432:5432") in published
    assert all(port.startswith("127.0.0.1:") for _, port in published), published
