from pydantic import BaseModel, conlist, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class StatsRequest(BaseModel):
    nutrition_input: conlist(float, min_items=9, max_items=9) = Field(
        ...,
        description="Target nutritional values: [calories, fat, sat_fat, cholesterol, sodium, carbs, fiber, sugar, protein]"
    )
    metric: str = Field("nutritional_mae", pattern="^(nutritional_mae|diversity_score)$")
    bmi: Optional[float] = Field(None, gt=0, le=100)
    goal: Optional[str] = Field(None, pattern="^(weight_loss|muscle_gain|maintenance)$")


class FeatureImportanceResponse(BaseModel):
    feature_importance: List[Dict[str, Any]]
    methodology: str
    timestamp: datetime

class StatisticsResponse(BaseModel):
    recommendation_count: int
    nutrition_statistics: Dict[str, Any]
    error_analysis: Dict[str, Dict[str, float]]
    diversity_score: float
    model_used: str
    target_nutrition: Dict[str, float]
    timestamp: datetime