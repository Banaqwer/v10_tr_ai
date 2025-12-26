"""
Pydantic-based configuration loader with support for environment variables and YAML files.
Provides a complete, production-ready settings management system.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Set
from enum import Enum

import yaml
from pydantic import BaseSettings, Field, validator, SecretStr
from pydantic.env_settings import SettingsSourceCallable
from dotenv import load_dotenv


# Configure logging
logger = logging.getLogger(__name__)


class EnvironmentEnum(str, Enum):
    """Supported environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    host: str = Field("localhost", description="Database host")
    port: int = Field(5432, description="Database port")
    username: str = Field("admin", description="Database username")
    password: SecretStr = Field(default="", description="Database password")
    database: str = Field("v10_tr_ai", description="Database name")
    driver: str = Field("postgresql", description="Database driver")
    pool_size: int = Field(10, description="Connection pool size")
    max_overflow: int = Field(20, description="Maximum overflow connections")
    
    @property
    def url(self) -> str:
        """Generate database connection URL."""
        password = self.password.get_secret_value() if self.password else ""
        if password:
            return f"{self.driver}://{self.username}:{password}@{self.host}:{self.port}/{self.database}"
        return f"{self.driver}://{self.host}:{self.port}/{self.database}"
    
    class Config:
        env_prefix = "DB_"
        env_file = ".env"
        case_sensitive = False


class RedisConfig(BaseSettings):
    """Redis configuration."""
    host: str = Field("localhost", description="Redis host")
    port: int = Field(6379, description="Redis port")
    database: int = Field(0, description="Redis database number")
    password: Optional[SecretStr] = Field(None, description="Redis password")
    ssl: bool = Field(False, description="Use SSL for Redis connection")
    ssl_certfile: Optional[str] = Field(None, description="SSL certificate file path")
    ssl_keyfile: Optional[str] = Field(None, description="SSL key file path")
    
    @property
    def url(self) -> str:
        """Generate Redis connection URL."""
        protocol = "rediss" if self.ssl else "redis"
        if self.password:
            password = self.password.get_secret_value()
            return f"{protocol}://:{password}@{self.host}:{self.port}/{self.database}"
        return f"{protocol}://{self.host}:{self.port}/{self.database}"
    
    class Config:
        env_prefix = "REDIS_"
        env_file = ".env"
        case_sensitive = False


class APIConfig(BaseSettings):
    """API configuration."""
    host: str = Field("0.0.0.0", description="API host")
    port: int = Field(8000, description="API port")
    workers: int = Field(4, description="Number of workers")
    reload: bool = Field(False, description="Auto-reload on code changes")
    debug: bool = Field(False, description="Debug mode")
    title: str = Field("V10 TR AI API", description="API title")
    version: str = Field("1.0.0", description="API version")
    description: str = Field("Advanced Turkish AI Processing API", description="API description")
    cors_origins: list = Field(["*"], description="CORS allowed origins")
    cors_credentials: bool = Field(True, description="CORS credentials allowed")
    cors_methods: list = Field(["*"], description="CORS allowed methods")
    cors_headers: list = Field(["*"], description="CORS allowed headers")
    
    class Config:
        env_prefix = "API_"
        env_file = ".env"
        case_sensitive = False


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    level: str = Field("INFO", description="Logging level")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    log_file: Optional[str] = Field(None, description="Log file path")
    max_bytes: int = Field(10485760, description="Max log file size (10MB)")
    backup_count: int = Field(5, description="Number of backup log files")
    
    class Config:
        env_prefix = "LOG_"
        env_file = ".env"
        case_sensitive = False


class AIModelConfig(BaseSettings):
    """AI Model configuration."""
    model_name: str = Field("bert-base-turkish-cased", description="Model name")
    model_path: Optional[str] = Field(None, description="Path to local model")
    tokenizer_name: str = Field("bert-base-turkish-cased", description="Tokenizer name")
    max_sequence_length: int = Field(512, description="Maximum sequence length")
    batch_size: int = Field(32, description="Batch size for inference")
    device: str = Field("cpu", description="Device to use (cpu/cuda/mps)")
    use_gpu: bool = Field(False, description="Use GPU for inference")
    precision: str = Field("float32", description="Model precision (float32/float16)")
    cache_models: bool = Field(True, description="Cache downloaded models")
    models_cache_dir: str = Field(".cache/models", description="Models cache directory")
    
    class Config:
        env_prefix = "MODEL_"
        env_file = ".env"
        case_sensitive = False


class SecurityConfig(BaseSettings):
    """Security configuration."""
    secret_key: SecretStr = Field(..., description="Secret key for signing")
    algorithm: str = Field("HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(30, description="Access token expiration (minutes)")
    refresh_token_expire_days: int = Field(7, description="Refresh token expiration (days)")
    password_min_length: int = Field(8, description="Minimum password length")
    hash_algorithm: str = Field("bcrypt", description="Password hashing algorithm")
    api_key_prefix: str = Field("sk_", description="API key prefix")
    
    class Config:
        env_prefix = "SECURITY_"
        env_file = ".env"
        case_sensitive = False


class Settings(BaseSettings):
    """Main application settings."""
    
    # Environment
    environment: EnvironmentEnum = Field(EnvironmentEnum.DEVELOPMENT, description="Environment type")
    debug: bool = Field(False, description="Global debug mode")
    
    # Sub-configurations
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    api: APIConfig = APIConfig()
    logging: LoggingConfig = LoggingConfig()
    model: AIModelConfig = AIModelConfig()
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # Application settings
    app_name: str = Field("V10 TR AI", description="Application name")
    version: str = Field("1.0.0", description="Application version")
    timezone: str = Field("UTC", description="Application timezone")
    max_upload_size: int = Field(104857600, description="Max upload size (100MB)")
    temp_dir: str = Field("./temp", description="Temporary directory")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        env_nested_delimiter = "__"
    
    @validator("environment", pre=True)
    def validate_environment(cls, v: str) -> EnvironmentEnum:
        """Validate and convert environment value."""
        if isinstance(v, EnvironmentEnum):
            return v
        try:
            return EnvironmentEnum(v.lower())
        except ValueError:
            raise ValueError(
                f"Invalid environment: {v}. "
                f"Must be one of {[e.value for e in EnvironmentEnum]}"
            )
    
    @validator("debug", pre=True, always=True)
    def set_debug_from_environment(cls, v: bool, values: Dict[str, Any]) -> bool:
        """Set debug mode based on environment if not explicitly set."""
        if v is False and "environment" in values:
            environment = values.get("environment")
            if isinstance(environment, EnvironmentEnum):
                return environment == EnvironmentEnum.DEVELOPMENT
        return v
    
    @property
    def is_development(self) -> bool:
        """Check if in development environment."""
        return self.environment == EnvironmentEnum.DEVELOPMENT
    
    @property
    def is_production(self) -> bool:
        """Check if in production environment."""
        return self.environment == EnvironmentEnum.PRODUCTION
    
    @property
    def is_testing(self) -> bool:
        """Check if in testing environment."""
        return self.environment == EnvironmentEnum.TESTING


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config or {}
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML configuration: {e}")
        raise


def settings_customise_sources(
    init_settings,
    env_settings,
    file_settings,
) -> SettingsSourceCallable:
    """
    Customize settings sources priority.
    Priority: init_settings > env_settings > file_settings
    
    Args:
        init_settings: Settings from initialization
        env_settings: Settings from environment variables
        file_settings: Settings from file (.env)
        
    Returns:
        Ordered settings sources
    """
    return (
        init_settings,
        env_settings,
        file_settings,
    )


class ConfigLoader:
    """Configuration loader with support for multiple sources."""
    
    _instance: Optional["ConfigLoader"] = None
    _settings: Optional[Settings] = None
    
    def __new__(cls) -> "ConfigLoader":
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize configuration loader."""
        if self._settings is None:
            self._load_settings()
    
    @classmethod
    def _load_settings(cls) -> None:
        """Load settings from all sources."""
        # Load .env file
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)
            logger.info("Loaded environment variables from .env")
        
        # Check for YAML config
        yaml_config_path = os.getenv("CONFIG_PATH", None)
        yaml_settings = {}
        
        if yaml_config_path:
            try:
                yaml_settings = load_yaml_config(yaml_config_path)
            except FileNotFoundError:
                logger.warning(f"YAML config file not found: {yaml_config_path}")
        
        # Merge YAML and environment settings
        settings_dict = {**yaml_settings, **os.environ}
        
        try:
            cls._settings = Settings(**settings_dict)
            logger.info(f"Settings loaded successfully for {cls._settings.environment.value} environment")
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            # Fall back to defaults
            cls._settings = Settings()
    
    @classmethod
    def get_settings(cls) -> Settings:
        """
        Get settings instance.
        
        Returns:
            Settings instance
        """
        if cls._instance is None:
            cls()
        return cls._settings
    
    @classmethod
    def reload_settings(cls) -> None:
        """Reload settings from sources."""
        cls._settings = None
        cls._instance = None
        cls()
    
    @classmethod
    def export_settings(cls, format: str = "json") -> str:
        """
        Export settings to specified format.
        
        Args:
            format: Output format (json/yaml)
            
        Returns:
            Formatted settings string
        """
        if cls._settings is None:
            cls()
        
        settings_dict = cls._settings.dict()
        
        if format.lower() == "yaml":
            return yaml.dump(settings_dict, default_flow_style=False)
        else:
            import json
            return json.dumps(settings_dict, indent=2, default=str)


# Singleton instance
_config_loader = ConfigLoader()


def get_settings() -> Settings:
    """
    Get global settings instance.
    
    Returns:
        Settings instance
        
    Example:
        from settings import get_settings
        settings = get_settings()
        db_url = settings.database.url
    """
    return ConfigLoader.get_settings()


# Initialize settings on module import
if __name__ == "__main__":
    # Example usage
    settings = get_settings()
    
    print("=" * 60)
    print("Application Settings")
    print("=" * 60)
    print(f"Environment: {settings.environment.value}")
    print(f"Debug: {settings.debug}")
    print(f"App: {settings.app_name} v{settings.version}")
    print(f"\nAPI Configuration:")
    print(f"  Host: {settings.api.host}")
    print(f"  Port: {settings.api.port}")
    print(f"\nDatabase Configuration:")
    print(f"  Host: {settings.database.host}")
    print(f"  Port: {settings.database.port}")
    print(f"  Database: {settings.database.database}")
    print(f"\nRedis Configuration:")
    print(f"  Host: {settings.redis.host}")
    print(f"  Port: {settings.redis.port}")
    print(f"\nAI Model Configuration:")
    print(f"  Model: {settings.model.model_name}")
    print(f"  Device: {settings.model.device}")
    print(f"  Batch Size: {settings.model.batch_size}")
    print("=" * 60)
