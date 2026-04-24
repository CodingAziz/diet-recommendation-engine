from pydantic import BaseModel, conlist, Field
from typing import Optional, Dict, Any
from datetime import datetime 

class ExplainIn(BaseModel):
    recipe_id: int = Field(..., ge=0, description="Recipe index or ID for explanation")
    nutrition_input: conlist(float, min_items=9, max_items=9) = Field(
        ...,
        description="Target nutritional values: [calories, fat, sat_fat, cholesterol, sodium, carbs, fiber, sugar, protein]"
    )
    bmi: Optional[float] = Field(None, gt=0, le=100, description="Body Mass Index for personalization")
    goal: Optional[str] = Field(None, pattern="^(weight_loss|muscle_gain|maintenance)$",
                              description="Fitness goal: weight_loss, muscle_gain, or maintenance")


class ExplanationResponse(BaseModel):
    recipe_id: int
    recipe_name: str
    explanation: Dict[str, Any]
    model_used: str
    confidence: float
    timestamp: datetime