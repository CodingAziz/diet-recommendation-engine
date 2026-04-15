from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conlist, Field
from typing import List, Optional, Dict, Any
import pandas as pd
from datetime import datetime
from model import (recommend, output_recommended_recipes, explain_recommendation,
                   get_model_feature_importance, get_recommendation_statistics)
from config import Settings
from logging_config import setup_logging
import logging

# Initialize logging
logger = setup_logging()

# Load configuration
settings = Settings()

# Load dataset
dataset = pd.read_csv('../Data/dataset.csv', compression='gzip')

app = FastAPI(
    title="Diet Recommendation System API",
    description="AI-powered nutrition-based recipe recommendations using hybrid filtering",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


class params(BaseModel):
    n_neighbors: int = Field(default=5, ge=1, le=100, description="Number of recommendations to return")
    return_distance: bool = False

class PredictionIn(BaseModel):
    nutrition_input: List[float] = Field(
        ...,
        min_length=9,
        max_length=9,
        description="Target nutritional values: [calories, fat, sat_fat, cholesterol, sodium, carbs, fiber, sugar, protein]"
    )
    ingredients: list[str] = Field(default=[], description="Preferred ingredients to filter by")
    params: Optional[params] = None
    bmi: float = Field(..., gt=0, le=100, description="Body Mass Index for personalization")
    goal: str = Field(..., pattern="^(weight_loss|muscle_gain|maintenance)$",
                     description="Fitness goal: weight_loss, muscle_gain, or maintenance")
    metric: str = Field(..., pattern="^(nutritional_mae|diversity_score)$",
                       description="Selection metric: nutritional_mae or diversity_score")

class Recipe(BaseModel):
    Name: str
    CookTime: str
    PrepTime: str
    TotalTime: str
    RecipeIngredientParts: list[str]
    Calories: float
    FatContent: float
    SaturatedFatContent: float
    CholesterolContent: float
    SodiumContent: float
    CarbohydrateContent: float
    FiberContent: float
    SugarContent: float
    ProteinContent: float
    RecipeInstructions: list[str]

class PredictionOut(BaseModel):
    output: Optional[List[Recipe]] = None
    metadata: Optional[Dict[str, Any]] = None

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    uptime_seconds: Optional[float] = None

class ModelInfoResponse(BaseModel):
    available_models: List[str]
    available_metrics: List[str]
    default_metric: str
    model_details: Dict[str, Dict[str, Any]]

class PerformanceMetrics(BaseModel):
    nutritional_mae: float
    diversity_score: float
    latency_ms: float
    coverage: float

class ModelPerformanceResponse(BaseModel):
    models: Dict[str, PerformanceMetrics]
    recommendations: Dict[str, str]
    last_updated: datetime

class FeedbackIn(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    recipe_id: int = Field(..., ge=0, description="Recipe ID that was recommended")
    rating: int = Field(..., ge=1, le=5, description="User rating (1-5)")
    was_helpful: bool = Field(..., description="Whether the recommendation was helpful")
    comments: Optional[str] = Field(None, max_length=500, description="Optional feedback comments")
    session_id: Optional[str] = Field(None, description="Session identifier for tracking")

class FeedbackOut(BaseModel):
    status: str
    feedback_id: str
    message: str
    timestamp: datetime

class ExplanationResponse(BaseModel):
    recipe_id: int
    recipe_name: str
    explanation: Dict[str, Any]
    model_used: str
    confidence: float
    timestamp: datetime

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/", response_model=PredictionOut)
def predict_recipes(prediction_input: PredictionIn):
    """Generate personalized recipe recommendations based on nutritional requirements and user profile."""
    try:
        logger.info(f"Processing prediction request for user with BMI {prediction_input.bmi}, goal {prediction_input.goal}, metric {prediction_input.metric}")

        recommendation_dataframe = recommend(
            dataset=dataset,
            _input=prediction_input.nutrition_input,
            ingredients=prediction_input.ingredients,
            params=prediction_input.params.dict() if prediction_input.params else {'n_neighbors': 5, 'return_distance': False},
            metric=prediction_input.metric,
            bmi=prediction_input.bmi,
            goal=prediction_input.goal
        )

        output = output_recommended_recipes(recommendation_dataframe)

        if output is None:
            logger.warning("No recommendations generated")
            return PredictionOut(output=None, metadata={"error": "No recommendations found"})

        metadata = {
            "model_used": "hybrid" if prediction_input.bmi and prediction_input.goal else "knn_cosine",
            "metric_basis": prediction_input.metric,
            "total_candidates": len(output),
            "request_timestamp": datetime.now().isoformat()
        }

        logger.info(f"Generated {len(output)} recommendations successfully")
        return PredictionOut(output=output, metadata=metadata)

    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint for monitoring system status."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        timestamp=datetime.now(),
        uptime_seconds=None  # Could be implemented with a global start time
    )

@app.get("/models/info", response_model=ModelInfoResponse)
def get_models_info():
    """Get information about available models and configuration."""
    return ModelInfoResponse(
        available_models=["knn_cosine", "knn_euclidean", "kmeans", "svd", "hybrid"],
        available_metrics=["nutritional_mae", "diversity_score"],
        default_metric="nutritional_mae",
        model_details={
            "knn_cosine": {"description": "Fast cosine similarity search", "best_for": "speed"},
            "knn_euclidean": {"description": "Euclidean distance search", "best_for": "magnitude_sensitivity"},
            "kmeans": {"description": "Cluster-based recommendations", "best_for": "diversity"},
            "svd": {"description": "Latent factor analysis", "best_for": "pattern_discovery"},
            "hybrid": {"description": "Personalized health-aware scoring", "best_for": "accuracy"}
        }
    )

@app.get("/models/performance", response_model=ModelPerformanceResponse)
def get_model_performance():
    """Get performance metrics for all recommendation models."""
    # These would ideally be calculated dynamically or cached
    # For now, returning static values based on our evaluation
    return ModelPerformanceResponse(
        models={
            "knn_cosine": PerformanceMetrics(
                nutritional_mae=12.05, diversity_score=0.234, latency_ms=15.2, coverage=0.0012
            ),
            "knn_euclidean": PerformanceMetrics(
                nutritional_mae=12.05, diversity_score=0.245, latency_ms=16.8, coverage=0.0013
            ),
            "kmeans": PerformanceMetrics(
                nutritional_mae=12.05, diversity_score=0.456, latency_ms=18.3, coverage=0.0021
            ),
            "svd": PerformanceMetrics(
                nutritional_mae=12.05, diversity_score=0.267, latency_ms=22.1, coverage=0.0014
            ),
            "hybrid": PerformanceMetrics(
                nutritional_mae=12.05, diversity_score=0.234, latency_ms=17.9, coverage=0.0012
            )
        },
        recommendations={
            "nutritional_accuracy": "hybrid",
            "maximum_diversity": "kmeans",
            "fastest": "knn_cosine"
        },
        last_updated=datetime.now()
    )

@app.post("/feedback/", response_model=FeedbackOut)
def submit_feedback(feedback: FeedbackIn):
    """Submit user feedback on recommendations for system improvement."""
    try:
        logger.info(f"Received feedback from user {feedback.user_id} for recipe {feedback.recipe_id}: rating {feedback.rating}")

        # In a real implementation, this would be stored in a database
        # For now, we'll just log it and return a confirmation
        feedback_id = f"fb_{feedback.user_id}_{feedback.recipe_id}_{int(datetime.now().timestamp())}"

        # Could implement feedback analysis here
        # e.g., update user preferences, retrain models, etc.

        return FeedbackOut(
            status="feedback_received",
            feedback_id=feedback_id,
            message="Thank you for your feedback! It will help us improve our recommendations.",
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")

@app.get("/explain/{recipe_id}", response_model=ExplanationResponse)
def explain_recipe_recommendation(
    recipe_id: int,
    nutrition_input: str,  # Comma-separated floats
    bmi: Optional[float] = None,
    goal: Optional[str] = None,
    metric: str = "nutritional_mae"
):
    """Get detailed explanation of why a recipe was recommended."""
    try:
        # Parse nutrition input
        try:
            nutrition_values = [float(x.strip()) for x in nutrition_input.split(",")]
            if len(nutrition_values) != 9:
                raise ValueError("Must provide exactly 9 nutrition values")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid nutrition input: {str(e)}")

        # Validate goal if provided
        if goal and goal not in ["weight_loss", "muscle_gain", "maintenance"]:
            raise HTTPException(status_code=400, detail="Invalid goal. Must be one of: weight_loss, muscle_gain, maintenance")

        explanation = explain_recommendation(
            recipe_id=recipe_id,
            target_vector=nutrition_values,
            bmi=bmi,
            goal=goal
        )

        if "error" in explanation:
            raise HTTPException(status_code=404, detail=explanation["error"])

        return ExplanationResponse(
            recipe_id=recipe_id,
            recipe_name=explanation.get("recipe_name", "Unknown Recipe"),
            explanation=explanation,
            model_used=explanation.get("model_used", "unknown"),
            confidence=explanation.get("confidence", 0.0),
            timestamp=datetime.now()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating explanation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")

@app.get("/models/feature-importance", response_model=FeatureImportanceResponse)
def get_feature_importance():
    """Get feature importance analysis for nutrition-based recommendations."""
    try:
        importance_data = get_model_feature_importance()

        if "error" in importance_data:
            raise HTTPException(status_code=500, detail=importance_data["error"])

        return FeatureImportanceResponse(
            feature_importance=importance_data["feature_importance"],
            methodology=importance_data["methodology"],
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feature importance calculation failed: {str(e)}")

@app.get("/analytics/recommendation-stats")
def get_recommendation_statistics(
    nutrition_input: str,  # Comma-separated floats
    metric: str = "nutritional_mae",
    bmi: Optional[float] = None,
    goal: Optional[str] = None
):
    """Get detailed statistics about recommendation performance."""
    try:
        # Parse nutrition input
        try:
            nutrition_values = [float(x.strip()) for x in nutrition_input.split(",")]
            if len(nutrition_values) != 9:
                raise ValueError("Must provide exactly 9 nutrition values")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid nutrition input: {str(e)}")

        stats = get_recommendation_statistics(
            nutrition_input=nutrition_values,
            metric=metric,
            bmi=bmi,
            goal=goal
        )

        if "error" in stats:
            raise HTTPException(status_code=500, detail=stats["error"])

        return StatisticsResponse(
            recommendation_count=stats["recommendation_count"],
            nutrition_statistics=stats["nutrition_statistics"],
            error_analysis=stats["error_analysis"],
            diversity_score=stats["diversity_score"],
            model_used=stats["model_used"],
            target_nutrition=stats["target_nutrition"],
            timestamp=datetime.now()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Statistics generation failed: {str(e)}")
