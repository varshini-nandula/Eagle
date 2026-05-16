"""
services/detection/zones.py

Zone definitions are now loaded from config/zones.yaml via ZoneConfigLoader.
Set ZONES_CONFIG_PATH env var to override the default config location.

Previously hardcoded DEFAULT_ZONES have been removed.
"""

import logging
from libs.config.zone_loader import ZoneConfigLoader

logger = logging.getLogger(__name__)

# Module-level singleton loader — starts hot-reload background thread
_loader = ZoneConfigLoader()
_loader.start()


def get_zones() -> list[dict]:
    """
    Return the current list of zone dicts loaded from YAML.
    Each zone has: name, polygon, alert_on_entry, color_hex.
    """
    return _loader.get_zones()


def get_camera_id() -> str | None:
    """Return the camera_id from the active zone config."""
    return _loader.get_camera_id()


# Convenience alias for code that previously referenced DEFAULT_ZONES directly
DEFAULT_ZONES = get_zones()
