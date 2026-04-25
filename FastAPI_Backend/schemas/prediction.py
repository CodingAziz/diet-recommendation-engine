from pydantic import BaseModel, Field, conlist
from typing import List, Optional, Dict, Any
from .recipe import Recipe


class Params(BaseModel):
    n_neighbors: int = Field(
        default=5, ge=1, le=100,
        description="Number of recommendations to return"
    )
    return_distance: bool = False


class PredictionIn(BaseModel):
    nutrition_input: conlist(float, min_items=9, max_items=9) = Field(
        ...,
        description="Target nutritional values: [calories, fat, sat_fat, cholesterol, sodium, carbs, fiber, sugar, protein]"
    )
    ingredients: Optional[List[str]] = Field(
        default=None,
        description="Preferred ingredients to filter by"
    )
    params: Optional[Params] = None
    bmi: float = Field(
        ...,
        gt=0,
        le=100,
        description="Body Mass Index for personalization"
    )
    goal: str = Field(
        ...,
        pattern="^(weight_loss|muscle_gain|maintenance)$",
        description="Fitness goal"
    )
    metric: str = Field(
        ...,
        pattern="^(nutritional_mae|diversity_score)$",
        description="Selection metric"
    )

class PredictionOut(BaseModel):
    output: Optional[List[Recipe]] = None
    metadata: Optional[Dict[str, Any]] = None