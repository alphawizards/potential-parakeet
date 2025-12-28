"""
Backend Configuration
=====================
Centralized configuration using Pydantic Settings.
Supports all data sources and rate limiting.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from pathlib import Path
from typing import List, Optional, Union

# Compute env file path before class definition
_ENV_FILE_PATH = Path(__file__).parent / ".env"


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Use pydantic v2 model_config
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE_PATH),
        case_sensitive=True,
        extra="ignore"
    )
    
    # Application
    APP_NAME: str = "Quant Trading Dashboard API"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = True
    
    # Server
    HOST: str = "0.0.0.0"  # nosec B104
    PORT: int = 8000
    
    # Database
    DATABASE_URL: str = "sqlite:///./data/trades.db"
    
    # CORS - stored as comma-separated string
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://localhost:8000"
    
    # Tiingo API (US Stocks, ETFs, Gold)
    TIINGO_API_KEY: str = ""
    TIINGO_IS_PREMIUM: bool = True
    TIINGO_RATE_LIMIT: int = 20000  # requests per hour
    
    # yFinance Settings (ASX, VIX, BTC)
    YFINANCE_CACHE_HOURS: int = 24
    
    # Pagination
    DEFAULT_PAGE_SIZE: int = 50
    MAX_PAGE_SIZE: int = 200
    
    # Rate Limiting
    RATE_LIMIT_DEFAULT: str = "60/minute"
    RATE_LIMIT_SCAN: str = "10/minute"
    RATE_LIMIT_DATA_REFRESH: str = "5/minute"
    
    # API Key for protected endpoints (optional)
    API_KEY: str = ""
    API_KEY_HASH: str = ""
    
    @property
    def cors_origins_list(self) -> List[str]:
        """Get CORS origins as list."""
        if isinstance(self.CORS_ORIGINS, str):
            return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]
        return self.CORS_ORIGINS


# Global settings instance
settings = Settings()


# Ensure data directory exists
DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Ensure cache directory exists
CACHE_DIR = Path(__file__).parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Ensure reports directory exists
REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
