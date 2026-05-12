import pytest
from libs.config.policy_loader import PolicyLoader

def test_invalid_yaml(tmp_path, monkeypatch):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text("zones: [unclosed")

    monkeypatch.setenv("POLICY_PATH", str(bad_file))

    loader = PolicyLoader()

    with pytest.raises(ValueError):
        loader.load_policy()