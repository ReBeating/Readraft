from pathlib import Path

from app.config import Settings


def test_default_brand_and_database(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("APP_NAME", raising=False)
    monkeypatch.delenv("APP_DATABASE_PATH", raising=False)
    monkeypatch.setenv("APP_DATA_DIR", str(tmp_path))

    settings = Settings.from_env()

    assert settings.app_name == "叙枢"
    assert settings.database_path == tmp_path / "xushu.db"
