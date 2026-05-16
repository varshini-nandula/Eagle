from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    policy_path: str = "policies/default.yaml"
    detector_model: str = "yolov8n.pt"
    detection_confidence_threshold: float = 0.45
    detector_device: str = "cpu"
    tracker_fps: float = 30
    tracker_max_age: int = 30
    tracker_n_init: int = 3
    tracker_max_cosine_distance: float = 0.4
    camera_id: str = "cam_01"

    reasoning_dwell_threshold_seconds: float = 5.0
    reasoning_cooldown_seconds: float = 5.0

    # Kafka Settings
    use_kafka: bool = False
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "track-events"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
