from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Environment-backed connection / API settings
    redis_url: str = "redis://localhost:6379"
    vlm_provider: str = "mock"
    llm_provider: str = "mock"
    ollama_host: str = "http://localhost:11434"

    # YOLO / detection settings (kept for backward compatibility alongside existing names)
    yolo_model: str = "yolov8n.pt"
    detection_confidence: float = 0.4
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Action classifier thresholds
    lingering_threshold_sec: float = 5.0
    movement_threshold_px: float = 10.0
    near_keypad_dist_px: float = 80.0
    keypad_center_x: int = 320
    keypad_center_y: int = 240
    policy_path: str = "policies/default.yaml"
    detector_model: str = "yolov8n.pt"
    detection_confidence_threshold: float = 0.45
    detector_device: str = "cpu"
    tracker_fps: float = 30
    tracker_max_age: int = 30
    tracker_n_init: int = 3
    tracker_max_cosine_distance: float = 0.4
    camera_id: str = "cam_01"

    # Action classifier settings
    lingering_threshold_sec: float = 5.0
    movement_threshold_px: float = 15.0
    near_keypad_dist_px: float = 75.0
    keypad_center_x: float = 500.0
    keypad_center_y: float = 500.0

    # Reasoning trigger settings
    reasoning_dwell_threshold_seconds: float = 5.0
    reasoning_cooldown_seconds: float = 5.0

    # New reasoning / alert settings
    reasoning_trigger_sec: float = 5.0
    ring_buffer_max: int = 50
    alert_dedup_window: int = 300
    snapshot_dir: str = "/tmp/eagle_snapshots"
    cors_origins: list[str] = ["http://localhost:5173"]  # Vite dev
    max_alerts_page: int = 50

    # Action classifier settings
    lingering_threshold_sec: float = 10.0
    movement_threshold_px: float = 5.0
    near_keypad_dist_px: float = 80.0
    keypad_center_x: float = 640.0
    keypad_center_y: float = 360.0

    # Kafka Settings
    use_kafka: bool = False
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic: str = "track-events"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
