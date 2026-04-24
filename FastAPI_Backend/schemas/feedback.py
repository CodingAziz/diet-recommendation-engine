from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class FeedbackIn(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    recipe_id: str = Field(..., description="Recipe ID or name that was recommended")
    rating: int = Field(..., ge=1, le=5, description="User rating (1-5)")
    was_helpful: bool = Field(..., description="Whether the recommendation was helpful")
    comments: Optional[str] = Field(None, max_length=500, description="Optional feedback comments")
    session_id: Optional[str] = Field(None, description="Session identifier for tracking")


class FeedbackOut(BaseModel):
    status: str
    feedback_id: str
    message: str
    timestamp: datetime