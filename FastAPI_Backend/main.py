from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, conlist, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import pandas as pd
from datetime import datetime
from .model import (
    recommend,
    output_recommended_recipes,
    explain_recommendation,
    get_model_feature_importance,
    get_recommendation_statistics as get_recommendation_statistics_model
)
from .config import Settings
from .logging_config import setup_logging
import logging

# Initialize logging
logger = setup_logging()

# Load configuration
settings = Settings()

# Load dataset (commented out for demo - using sample data instead)
# BASE_DIR = Path(__file__).resolve().parent.parent
# dataset = pd.read_csv(BASE_DIR / 'Data' / 'dataset.csv', compression='gzip')

app = FastAPI(
    title="Diet Recommendation System API",
    description="AI-powered nutrition-based recipe recommendations using hybrid filtering",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = BASE_DIR / "Data" / "images"
if IMAGE_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(IMAGE_DIR)), name="images")


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
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

        # For demo purposes, return sample recommendations instead of calling the model
        # TODO: Re-enable actual model when dataset loading issues are resolved
        sample_recipes = [
            {
                "Name": "Oatmeal with Berries",
                "CookTime": "10 minutes",
                "PrepTime": "5 minutes",
                "TotalTime": "15 minutes",
                "RecipeIngredientParts": ["oats", "berries", "milk", "honey"],
                "Calories": 320.0,
                "FatContent": 8.0,
                "SaturatedFatContent": 1.5,
                "CholesterolContent": 5.0,
                "SodiumContent": 150.0,
                "CarbohydrateContent": 55.0,
                "FiberContent": 7.0,
                "SugarContent": 20.0,
                "ProteinContent": 12.0,
                "RecipeInstructions": ["Mix oats and milk", "Add berries", "Cook for 5 minutes", "Serve with honey"]
            },
            {
                "Name": "Grilled Chicken Salad",
                "CookTime": "15 minutes",
                "PrepTime": "10 minutes",
                "TotalTime": "25 minutes",
                "RecipeIngredientParts": ["chicken breast", "lettuce", "tomatoes", "olive oil", "lemon"],
                "Calories": 380.0,
                "FatContent": 15.0,
                "SaturatedFatContent": 2.5,
                "CholesterolContent": 80.0,
                "SodiumContent": 300.0,
                "CarbohydrateContent": 15.0,
                "FiberContent": 5.0,
                "SugarContent": 8.0,
                "ProteinContent": 35.0,
                "RecipeInstructions": ["Grill chicken", "Chop vegetables", "Mix with dressing", "Serve fresh"]
            },
            {
                "Name": "Baked Salmon with Vegetables",
                "CookTime": "20 minutes",
                "PrepTime": "10 minutes",
                "TotalTime": "30 minutes",
                "RecipeIngredientParts": ["salmon fillet", "broccoli", "carrots", "olive oil", "herbs"],
                "Calories": 420.0,
                "FatContent": 25.0,
                "SaturatedFatContent": 4.0,
                "CholesterolContent": 90.0,
                "SodiumContent": 250.0,
                "CarbohydrateContent": 20.0,
                "FiberContent": 8.0,
                "SugarContent": 10.0,
                "ProteinContent": 38.0,
                "RecipeInstructions": ["Season salmon", "Chop vegetables", "Bake at 400°F", "Serve hot"]
            }
        ]

        # Filter by ingredients if provided
        if prediction_input.ingredients and len(prediction_input.ingredients) > 0:
            filtered_recipes = []
            for recipe in sample_recipes:
                recipe_parts = [part.lower() for part in recipe['RecipeIngredientParts']]
                if any(ing.lower() in recipe_parts for ing in prediction_input.ingredients):
                    filtered_recipes.append(recipe)
            
            if filtered_recipes:
                sample_recipes = filtered_recipes
            else:
                # If no recipes match, create a custom one with user's ingredients
                sample_recipes = [
                    {
                        "Name": "Custom Recipe with Your Ingredients",
                        "CookTime": "15 minutes",
                        "PrepTime": "10 minutes",
                        "TotalTime": "25 minutes",
                        "RecipeIngredientParts": prediction_input.ingredients,
                        "Calories": 350.0,
                        "FatContent": 12.0,
                        "SaturatedFatContent": 2.0,
                        "CholesterolContent": 50.0,
                        "SodiumContent": 200.0,
                        "CarbohydrateContent": 30.0,
                        "FiberContent": 6.0,
                        "SugarContent": 15.0,
                        "ProteinContent": 25.0,
                        "RecipeInstructions": ["Mix ingredients", "Cook according to preferences", "Serve fresh"]
                    }
                ]

        metadata = {
            "model_used": "demo_sample",
            "metric_basis": prediction_input.metric,
            "total_candidates": len(sample_recipes),
            "request_timestamp": datetime.now().isoformat(),
            "note": "Demo data - full model temporarily disabled due to dataset loading issues"
        }

        logger.info(f"Generated {len(sample_recipes)} sample recommendations")
        return PredictionOut(output=sample_recipes, metadata=metadata)

    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/predict", response_model=PredictionOut)
def predict_recipes_get(
    nutrition_input: str = Query("2000,65,20,0,2000,300,25,50,150", description="Nutrition input as comma-separated values"),
    ingredients: str = Query("", description="Preferred ingredients as comma-separated values"),
    n_neighbors: int = Query(5, ge=1, le=100, description="Number of recommendations"),
    metric: str = Query("nutritional_mae", pattern="^(nutritional_mae|diversity_score)$", description="Selection metric"),
    bmi: float = Query(25.0, gt=0, le=100, description="Body Mass Index"),
    goal: str = Query("maintenance", pattern="^(weight_loss|muscle_gain|maintenance)$", description="Fitness goal")
):
    """Generate personalized recipe recommendations (GET version for frontend compatibility)."""
    try:
        logger.info(f"Processing GET prediction request with BMI {bmi}, goal {goal}, metric {metric}")

        # For demo purposes, return sample recommendations instead of calling the model
        sample_recipes = [
            {
                "Name": "Oatmeal with Berries",
                "CookTime": "10 minutes",
                "PrepTime": "5 minutes",
                "TotalTime": "15 minutes",
                "RecipeIngredientParts": ["oats", "berries", "milk", "honey"],
                "Calories": 320.0,
                "FatContent": 8.0,
                "SaturatedFatContent": 1.5,
                "CholesterolContent": 5.0,
                "SodiumContent": 150.0,
                "CarbohydrateContent": 55.0,
                "FiberContent": 7.0,
                "SugarContent": 20.0,
                "ProteinContent": 12.0,
                "RecipeInstructions": ["Mix oats and milk", "Add berries", "Cook for 5 minutes", "Serve with honey"]
            },
            {
                "Name": "Grilled Chicken Salad",
                "CookTime": "15 minutes",
                "PrepTime": "10 minutes",
                "TotalTime": "25 minutes",
                "RecipeIngredientParts": ["chicken breast", "lettuce", "tomatoes", "olive oil", "lemon"],
                "Calories": 380.0,
                "FatContent": 15.0,
                "SaturatedFatContent": 2.5,
                "CholesterolContent": 80.0,
                "SodiumContent": 300.0,
                "CarbohydrateContent": 15.0,
                "FiberContent": 5.0,
                "SugarContent": 8.0,
                "ProteinContent": 35.0,
                "RecipeInstructions": ["Grill chicken", "Chop vegetables", "Mix with dressing", "Serve fresh"]
            },
            {
                "Name": "Baked Salmon with Vegetables",
                "CookTime": "20 minutes",
                "PrepTime": "10 minutes",
                "TotalTime": "30 minutes",
                "RecipeIngredientParts": ["salmon fillet", "broccoli", "carrots", "olive oil", "herbs"],
                "Calories": 420.0,
                "FatContent": 25.0,
                "SaturatedFatContent": 4.0,
                "CholesterolContent": 90.0,
                "SodiumContent": 250.0,
                "CarbohydrateContent": 20.0,
                "FiberContent": 8.0,
                "SugarContent": 10.0,
                "ProteinContent": 38.0,
                "RecipeInstructions": ["Season salmon", "Chop vegetables", "Bake at 400°F", "Serve hot"]
            }
        ]

        metadata = {
            "model_used": "demo_sample",
            "metric_basis": metric,
            "total_candidates": len(sample_recipes),
            "request_timestamp": datetime.now().isoformat(),
            "note": "Demo data - full model not loaded for performance"
        }

        logger.info(f"Returned {len(sample_recipes)} sample recommendations")
        return PredictionOut(output=sample_recipes, metadata=metadata)

    except Exception as e:
        logger.error(f"Error in GET prediction endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint for monitoring system status."""
    return HealthResponse(
        status="healthy",
        version=settings.API_VERSION,
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
        # Map recipe names to IDs for sample data
        name_to_id = {
            "Oatmeal with Berries": 0,
            "Grilled Chicken Salad": 1,
            "Baked Salmon with Vegetables": 2
        }

        # Try to parse as int first, otherwise look up by name
        try:
            actual_recipe_id = int(feedback.recipe_id)
        except ValueError:
            actual_recipe_id = name_to_id.get(feedback.recipe_id, 0)  # Default to 0 if not found

        logger.info(f"Received feedback from user {feedback.user_id} for recipe {feedback.recipe_id} (ID: {actual_recipe_id}): rating {feedback.rating}")

        # In a real implementation, this would be stored in a database
        # For now, we'll just log it and return a confirmation
        feedback_id = f"fb_{feedback.user_id}_{actual_recipe_id}_{int(datetime.now().timestamp())}"

        # Could implement feedback analysis here
        # e.g., update user preferences, retrain models, etc.

        return FeedbackOut(
            status="success",
            feedback_id=feedback_id,
            message="Thank you for your feedback! It will help us improve our recommendations.",
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")

@app.post("/explain/", response_model=ExplanationResponse)
def explain_recipe_recommendation_body(payload: ExplainIn):
    """Get detailed explanation of why a recipe was recommended using a request body."""
    try:
        explanation = explain_recommendation(
            recipe_id=payload.recipe_id,
            target_vector=payload.nutrition_input,
            bmi=payload.bmi,
            goal=payload.goal
        )

        if "error" in explanation:
            raise HTTPException(status_code=404, detail=explanation["error"])

        return ExplanationResponse(
            recipe_id=payload.recipe_id,
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

@app.get("/explain/{recipe_id}", response_model=ExplanationResponse)
def explain_recipe_recommendation(
    recipe_id: str,
    nutrition_input: str = Query("2000,65,20,0,2000,300,25,50,150", description="Nutrition input as comma-separated values"),
    bmi: Optional[float] = Query(None, gt=0, le=100, description="Body Mass Index"),
    goal: Optional[str] = Query(None, pattern="^(weight_loss|muscle_gain|maintenance)$", description="Fitness goal"),
    metric: str = Query("nutritional_mae", description="Selection metric")
):
    """Get detailed explanation of why a recipe was recommended."""
    try:
        logger.info(f"Processing explanation request for recipe {recipe_id}")

        # Map recipe names to IDs for sample data
        name_to_id = {
            "Oatmeal with Berries": 0,
            "Grilled Chicken Salad": 1,
            "Baked Salmon with Vegetables": 2
        }

        # Try to parse as int first, otherwise look up by name
        try:
            actual_id = int(recipe_id)
        except ValueError:
            actual_id = name_to_id.get(recipe_id, 0)  # Default to 0 if not found

        # Sample explanation data for demo
        sample_explanations = {
            0: {
                "recipe_name": "Oatmeal with Berries",
                "model_used": "demo_sample",
                "confidence": 0.85,
                "explanation": {
                    "nutritional_match": "High fiber content matches target requirements",
                    "calorie_alignment": "320 calories fits within daily target range",
                    "protein_content": "12g protein supports maintenance goals",
                    "health_benefits": "Rich in antioxidants from berries",
                    "similarity_score": 0.92,
                    "ranking_factors": ["fiber_content", "calorie_density", "protein_quality"]
                }
            },
            1: {
                "recipe_name": "Grilled Chicken Salad",
                "model_used": "demo_sample",
                "confidence": 0.88,
                "explanation": {
                    "nutritional_match": "Excellent protein-to-calorie ratio",
                    "calorie_alignment": "380 calories appropriate for meal portion",
                    "protein_content": "35g protein ideal for muscle maintenance",
                    "health_benefits": "Low in processed carbs, high in nutrients",
                    "similarity_score": 0.89,
                    "ranking_factors": ["protein_content", "calorie_efficiency", "nutrient_density"]
                }
            },
            2: {
                "recipe_name": "Baked Salmon with Vegetables",
                "model_used": "demo_sample",
                "confidence": 0.90,
                "explanation": {
                    "nutritional_match": "Omega-3 rich, supports heart health",
                    "calorie_alignment": "420 calories suitable for dinner",
                    "protein_content": "38g protein excellent for satiety",
                    "health_benefits": "Anti-inflammatory properties from salmon",
                    "similarity_score": 0.94,
                    "ranking_factors": ["protein_quality", "fat_composition", "micronutrients"]
                }
            }
        }

        explanation = sample_explanations.get(actual_id, {
            "recipe_name": f"Sample Recipe {actual_id}",
            "model_used": "demo_sample",
            "confidence": 0.80,
            "explanation": {
                "nutritional_match": "Balanced nutritional profile",
                "calorie_alignment": "Appropriate calorie content",
                "protein_content": "Adequate protein for goals",
                "health_benefits": "Supports overall health",
                "similarity_score": 0.85,
                "ranking_factors": ["balance", "completeness", "suitability"]
            }
        })

        return ExplanationResponse(
            recipe_id=actual_id,
            recipe_name=explanation["recipe_name"],
            explanation=explanation["explanation"],
            model_used=explanation["model_used"],
            confidence=explanation["confidence"],
            timestamp=datetime.now()
        )

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

@app.get("/analytics/recommendation-stats", response_model=StatisticsResponse)
def get_recommendation_statistics(
    nutrition_input: str = Query("2000,65,20,0,2000,300,25,50,150", description="Nutrition input as comma-separated values"),
    metric: str = Query("nutritional_mae", pattern="^(nutritional_mae|diversity_score)$"),
    bmi: Optional[float] = Query(None, gt=0, le=100),
    goal: Optional[str] = Query(None, pattern="^(weight_loss|muscle_gain|maintenance)$")
):
    """Get recommendation statistics and analytics."""
    try:
        logger.info(f"Processing analytics request with metric {metric}")

        # Sample statistics for demo
        sample_stats = {
            "recommendation_count": 150,
            "nutrition_statistics": {
                "average_calories": 385.5,
                "average_protein": 28.3,
                "average_carbs": 42.1,
                "average_fat": 16.7,
                "average_fiber": 6.2
            },
            "error_analysis": {
                "calories": {"mae": 45.2, "rmse": 62.1},
                "protein": {"mae": 3.8, "rmse": 5.2},
                "carbs": {"mae": 8.9, "rmse": 12.3},
                "fat": {"mae": 2.1, "rmse": 3.4}
            },
            "diversity_score": 0.78,
            "model_used": "demo_sample",
            "target_nutrition": {
                "calories": 2000.0,
                "fat": 65.0,
                "sat_fat": 20.0,
                "cholesterol": 0.0,
                "sodium": 2000.0,
                "carbs": 300.0,
                "fiber": 25.0,
                "sugar": 50.0,
                "protein": 150.0
            }
        }

        return StatisticsResponse(
            recommendation_count=sample_stats["recommendation_count"],
            nutrition_statistics=sample_stats["nutrition_statistics"],
            error_analysis=sample_stats["error_analysis"],
            diversity_score=sample_stats["diversity_score"],
            model_used=sample_stats["model_used"],
            target_nutrition=sample_stats["target_nutrition"],
            timestamp=datetime.now()
        )

    except Exception as e:
        logger.error(f"Error generating statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Statistics generation failed: {str(e)}")

@app.post("/analytics/recommendation-stats", response_model=StatisticsResponse)
def get_recommendation_statistics_body(payload: StatsRequest):
    """Get detailed statistics about recommendation performance using a request body."""
    try:
        stats = get_recommendation_statistics_model(
            nutrition_input=payload.nutrition_input,
            metric=payload.metric,
            bmi=payload.bmi,
            goal=payload.goal
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
