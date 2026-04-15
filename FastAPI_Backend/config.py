"""Configuration management for Diet Recommendation System"""
import os
from typing import Optional


class Settings:
    """Application settings"""
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE", None)
    
    # Model Configuration
    K_CANDIDATES: int = 50
    TOP_K: int = 10
    N_CLUSTERS: int = 20
    N_COMPONENTS: int = 5
    RANDOM_SEED: int = 42
    
    # Data Configuration
    DATA_PATH: str = os.getenv("DATA_PATH", "../Data/dataset.csv")
    CACHE_ENABLED: bool = os.getenv("CACHE_ENABLED", "True").lower() == "true"
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour in seconds
    
    # API Configuration
    API_TITLE: str = "Diet Recommendation System API"
    API_DESCRIPTION: str = "AI-powered nutrition-based recipe recommendations using hybrid filtering"
    API_VERSION: str = "1.0.0"
    
    # Validation
    MIN_CALORIES: float = 0.0
    MAX_CALORIES: float = 5000.0
    MIN_MACRO_VALUE: float = 0.0
    MAX_FAT: float = 200.0
    MAX_SODIUM: float = 10000.0
    
    # Valid options
    VALID_GOALS: list = ["weight_loss", "muscle_gain", "maintenance"]
    VALID_METRICS: list = ["nutritional_mae", "diversity_score"]
    
    # Performance
    DEFAULT_N_NEIGHBORS: int = 5
    MAX_N_NEIGHBORS: int = 100
    
    @classmethod
    def get_settings(cls) -> "Settings":
        """Get settings instance"""
        return cls()
    
    @classmethod
    def to_dict(cls) -> dict:
        """Convert settings to dictionary"""
        return {
            key: getattr(cls, key)
            for key in dir(cls)
            if not key.startswith('_') and key.isupper()
        }


settings = Settings.get_settings()
