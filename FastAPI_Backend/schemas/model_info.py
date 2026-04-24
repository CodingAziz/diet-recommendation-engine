from pydantic import BaseModel
from typing import List, Dict, Any

class ModelInfoResponse(BaseModel):
    available_models: List[str]
    available_metrics: List[str]
    default_metric: str
    model_details: Dict[str, Dict[str, Any]]