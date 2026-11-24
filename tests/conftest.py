import pytest


@pytest.fixture(autouse=True)
def _isolate_user_environment(tmp_path_factory, monkeypatch):
    """Keep the developer's own config, cache and Plex/Jellyfin credentials out of tests.

    Without this, ~/.config/audio-censor/config.toml overrides the built-in
    defaults the tests assert against (e.g. a personal ``mode = "beep"``).
    """
    root = tmp_path_factory.mktemp("xdg")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(root / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(root / "cache"))
    monkeypatch.delenv("PLEX_URL", raising=False)
    monkeypatch.delenv("PLEX_TOKEN", raising=False)
    monkeypatch.delenv("JELLYFIN_URL", raising=False)
    monkeypatch.delenv("JELLYFIN_API_KEY", raising=False)
