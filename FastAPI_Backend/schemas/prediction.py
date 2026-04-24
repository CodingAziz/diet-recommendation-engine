from pydantic import BaseModel, Field, conlist
from typing import Optional, List, Dict, Any
from .recipe import Recipe 

class params(BaseModel):
    n_neighbors: int = Field(default=5, ge=1, le=100, description="Number of recommendations to return")
    return_distance: bool = False

class PredictionIn(BaseModel):
    nutrition_input: conlist(float, min_items=9, max_items=9) = Field(
        ...,
        description="Target nutritional values: [calories, fat, sat_fat, cholesterol, sodium, carbs, fiber, sugar, protein]"
    )
    ingredients: list[str] = Field(default=[], description="Preferred ingredients to filter by")
    params: Optional[params] = None
    bmi: float = Field(..., gt=0, le=100, description="Body Mass Index for personalization")
    goal: str = Field(..., pattern="^(weight_loss|muscle_gain|maintenance)$",
                     description="Fitness goal: weight_loss, muscle_gain, or maintenance")
    metric: str = Field(..., pattern="^(nutritional_mae|diversity_score)$",
                       description="Selection metric: nutritional_mae or diversity_score")

class PredictionOut(BaseModel):
    output: Optional[List[Recipe]] = None
    metadata: Optional[Dict[str, Any]] = None