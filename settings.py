"""
Configuration loader using Pydantic BaseSettings.
Supports loading configuration from environment variables and YAML files.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings configuration loader.
    
    Supports configuration from:
    1. Environment variables (prefixed with APP_)
    2. YAML configuration files
    3. Default values
    
    Environment variables take precedence over YAML settings.
    """
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    api_debug: bool = Field(default=False, description="Enable debug mode")
    
    # Database Configuration
    database_url: str = Field(
        default="sqlite:///./app.db",
        description="Database connection URL"
    )
    database_echo: bool = Field(
        default=False,
        description="Enable SQL echo logging"
    )
    
    # Authentication
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for JWT tokens"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration time in minutes"
    )
    
    # Application
    app_name: str = Field(default="v10_tr_ai", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(default="development", description="Environment (development/production/testing)")
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default=None, description="Log file path (optional)")
    
    # AI/ML Configuration
    model_name: str = Field(default="gpt-3.5-turbo", description="AI model name")
    model_temperature: float = Field(default=0.7, description="Model temperature")
    max_tokens: int = Field(default=2000, description="Maximum tokens for generation")
    
    # Optional service configurations
    redis_url: Optional[str] = Field(default=None, description="Redis connection URL")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="APP_",
        case_sensitive=False,
        extra="ignore",
    )
    
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "Settings":
        """
        Load settings from a YAML configuration file.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            Settings instance with values from YAML and environment variables
            
        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
        
        # Environment variables override YAML settings
        env_vars = {
            key.replace(f"{cls.model_config['env_prefix']}", "", 1).lower(): value
            for key, value in os.environ.items()
            if key.startswith(cls.model_config['env_prefix'])
        }
        
        # Merge YAML and environment variables (env vars take precedence)
        config_data = {**yaml_data, **env_vars}
        
        return cls(**config_data)
    
    @classmethod
    def from_yaml_or_env(
        cls,
        yaml_path: Optional[str | Path] = None,
        default_yaml_path: str | Path = "config.yaml",
    ) -> "Settings":
        """
        Load settings from YAML file if it exists, otherwise from environment.
        
        Args:
            yaml_path: Explicit path to YAML configuration file
            default_yaml_path: Default path to check for YAML configuration
            
        Returns:
            Settings instance
        """
        # Use provided path or check default
        config_file = yaml_path or default_yaml_path
        config_file = Path(config_file)
        
        if config_file.exists():
            return cls.from_yaml(config_file)
        
        # Fall back to environment variables and defaults
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to dictionary.
        
        Returns:
            Dictionary representation of settings
        """
        return self.model_dump()
    
    def save_to_yaml(self, yaml_path: str | Path) -> None:
        """
        Save current settings to a YAML file.
        
        Args:
            yaml_path: Path where to save the YAML configuration
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def __repr__(self) -> str:
        """String representation of settings (masks sensitive values)."""
        safe_dict = self.model_dump()
        # Mask sensitive values
        sensitive_keys = {"secret_key", "openai_api_key"}
        for key in sensitive_keys:
            if key in safe_dict and safe_dict[key]:
                safe_dict[key] = "***MASKED***"
        return f"Settings({safe_dict})"


# Global settings instance (lazy loaded)
_settings: Optional[Settings] = None


def get_settings(yaml_path: Optional[str | Path] = None) -> Settings:
    """
    Get the global settings instance.
    Loads from YAML if available, otherwise from environment variables.
    
    Args:
        yaml_path: Optional path to YAML configuration file
        
    Returns:
        Settings instance
    """
    global _settings
    
    if _settings is None:
        _settings = Settings.from_yaml_or_env(yaml_path=yaml_path)
    
    return _settings


def reload_settings(yaml_path: Optional[str | Path] = None) -> Settings:
    """
    Reload settings (useful for testing or dynamic configuration updates).
    
    Args:
        yaml_path: Optional path to YAML configuration file
        
    Returns:
        New Settings instance
    """
    global _settings
    _settings = Settings.from_yaml_or_env(yaml_path=yaml_path)
    return _settings


# Example usage and documentation
if __name__ == "__main__":
    # Example 1: Load from environment variables and defaults
    settings = get_settings()
    print("Settings from environment and defaults:")
    print(settings)
    print()
    
    # Example 2: Load from YAML file
    # settings = Settings.from_yaml("config.yaml")
    # print("Settings from YAML:")
    # print(settings)
    # print()
    
    # Example 3: Load from YAML with environment variable overrides
    # APP_API_PORT=9000 python settings.py
    # This will use port 9000 instead of the value in config.yaml
    
    # Example 4: Save settings to YAML
    # settings.save_to_yaml("config_output.yaml")
