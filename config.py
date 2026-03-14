"""
Flask application configuration.

Usage:
    Set FLASK_ENV=production for production settings.
    Defaults to development if not set.
"""

import os


class BaseConfig:
    """Shared configuration across all environments."""
    TESTING = False
    CORS_ORIGINS = os.environ.get(
        "CORS_ORIGINS", "http://localhost:3000,http://localhost:3001"
    ).split(",")


class DevelopmentConfig(BaseConfig):
    """Development configuration — debug enabled, permissive CORS."""
    DEBUG = True


class ProductionConfig(BaseConfig):
    """Production configuration — debug off, restricted CORS."""
    DEBUG = False
    CORS_ORIGINS = os.environ.get(
        "CORS_ORIGINS",
        "https://biocs-app.vercel.app"
    ).split(",")


class TestingConfig(BaseConfig):
    """Testing configuration."""
    TESTING = True
    DEBUG = False


_config_map = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}


def get_config():
    """Return the config class based on FLASK_ENV environment variable."""
    env = os.environ.get("FLASK_ENV", "development").lower()
    return _config_map.get(env, DevelopmentConfig)
