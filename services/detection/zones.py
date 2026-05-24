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


def get_zones_for_point(x: float, y: float) -> list:
    """
    Return all zones whose polygon contains the point (x, y).

    Used by tracker.py to determine zone membership of a tracked object
    given its centre-point coordinates.

    Each returned object exposes a `.name` attribute so callers can do:
        zones = [z.name for z in get_zones_for_point(cx, cy)]

    Falls back gracefully (returns empty list) when:
    - No zones are configured yet
    - A zone polygon is malformed
    - shapely is not installed (point-in-polygon skipped)
    """
    zones = _loader.get_zones()
    if not zones:
        return []

    matched = []

    try:
        from shapely.geometry import Point, Polygon

        point = Point(x, y)
        for zone in zones:
            try:
                poly = Polygon(zone["polygon"])
                if poly.contains(point):
                    # Return a lightweight object with .name attribute
                    matched.append(_Zone(zone["name"]))
            except Exception as e:
                logger.warning(
                    "Skipping malformed polygon for zone '%s': %s",
                    zone.get("name", "unknown"),
                    e,
                )

    except ImportError:
        # shapely not installed — skip spatial check, return empty
        logger.debug(
            "shapely not available; get_zones_for_point returning [] for (%.1f, %.1f)",
            x, y,
        )

    return matched


class _Zone:
    """Lightweight zone result object exposing just the .name attribute."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        """Store the zone name."""
        self.name = name

    def __repr__(self) -> str:
        """Return a readable string representation of the zone."""
        return f"Zone(name={self.name!r})"

# Convenience alias for code that previously referenced DEFAULT_ZONES directly
DEFAULT_ZONES = get_zones()
