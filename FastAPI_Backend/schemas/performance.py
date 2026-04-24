from pydantic import BaseModel
from typing import Dict
from datetime import datetime

class PerformanceMetrics(BaseModel):
    nutritional_mae: float
    diversity_score: float
    latency_ms: float
    coverage: float

class ModelPerformanceResponse(BaseModel):
    models: Dict[str, PerformanceMetrics]
    recommendations: Dict[str, str]
    last_updated: datetime