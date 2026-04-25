from fastapi import APIRouter, HTTPException

from ..schemas.prediction import PredictionIn, PredictionOut
from ..services.recommendation_services import RecommendationService

router = APIRouter(prefix="/predictions", tags=["Predictions"])

service = RecommendationService()

@router.post("/predict", response_model=PredictionOut)
def predict_recipes(prediction_input: PredictionIn):
    try:
        results, model_used = service.predict(
            input_vec=prediction_input.nutrition_input,
            metric=prediction_input.metric,
            bmi=prediction_input.bmi,
            goal=prediction_input.goal,
            top_k=50,
            ingredients=prediction_input.ingredients
        )

        return PredictionOut(
            output=results.to_dict(orient="records"),
            metadata={
                "model_used": model_used,
                "count": len(results)
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))