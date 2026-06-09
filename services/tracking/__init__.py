"""services.tracking — Detection + tracking layer.

Heavy dependencies (cv2, ultralytics) are imported lazily by the individual
modules so that lightweight consumers (tests, memory service) can import
sub-modules like ``cross_camera_reid`` without pulling in the full stack.
"""
__all__: list[str] = []
