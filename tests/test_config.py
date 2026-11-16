from intent_agent.config import Settings


def test_settings_load_debug_and_respect_environment_precedence(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("DEBUG=false\n")
    monkeypatch.setenv("DEBUG", "true")

    settings = Settings(_env_file=env_file)

    assert settings.debug is True


def test_settings_default_debug_is_false(tmp_path):
    settings = Settings(_env_file=tmp_path / "missing.env")

    assert settings.debug is False
