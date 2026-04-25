from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from .config import Settings
from .logging_config import setup_logging
from .routes import predictions

# Initialize logging
logger = setup_logging()

# Load configuration
settings = Settings()

app = FastAPI(
    title="Diet Recommendation System API",
    description="AI-powered nutrition-based recipe recommendations using hybrid filtering",
    version="1.0.0",
    docs_url="/docs",
)

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

app.include_router(predictions.router)
    
# @app.get("/health", response_model=HealthResponse)
# def health_check():
#     """Health check endpoint for monitoring system status."""
#     return HealthResponse(
#         status="healthy",
#         version=settings.API_VERSION,
#         timestamp=datetime.now(),
#         uptime_seconds=None  # Could be implemented with a global start time
#     )

# @app.get("/models/info", response_model=ModelInfoResponse)
# def get_models_info():
#     """Get information about available models and configuration."""
#     return ModelInfoResponse(
#         available_models=["knn_cosine", "knn_euclidean", "kmeans", "svd", "hybrid"],
#         available_metrics=["nutritional_mae", "diversity_score"],
#         default_metric="nutritional_mae",
#         model_details={
#             "knn_cosine": {"description": "Fast cosine similarity search", "best_for": "speed"},
#             "knn_euclidean": {"description": "Euclidean distance search", "best_for": "magnitude_sensitivity"},
#             "kmeans": {"description": "Cluster-based recommendations", "best_for": "diversity"},
#             "svd": {"description": "Latent factor analysis", "best_for": "pattern_discovery"},
#             "hybrid": {"description": "Personalized health-aware scoring", "best_for": "accuracy"}
#         }
#     )

# @app.get("/models/performance", response_model=ModelPerformanceResponse)
# def get_model_performance():
#     """Get performance metrics for all recommendation models."""
#     # These would ideally be calculated dynamically or cached
#     # For now, returning static values based on our evaluation
#     return ModelPerformanceResponse(
#         models={
#             "knn_cosine": PerformanceMetrics(
#                 nutritional_mae=12.05, diversity_score=0.234, latency_ms=15.2, coverage=0.0012
#             ),
#             "knn_euclidean": PerformanceMetrics(
#                 nutritional_mae=12.05, diversity_score=0.245, latency_ms=16.8, coverage=0.0013
#             ),
#             "kmeans": PerformanceMetrics(
#                 nutritional_mae=12.05, diversity_score=0.456, latency_ms=18.3, coverage=0.0021
#             ),
#             "svd": PerformanceMetrics(
#                 nutritional_mae=12.05, diversity_score=0.267, latency_ms=22.1, coverage=0.0014
#             ),
#             "hybrid": PerformanceMetrics(
#                 nutritional_mae=12.05, diversity_score=0.234, latency_ms=17.9, coverage=0.0012
#             )
#         },
#         recommendations={
#             "nutritional_accuracy": "hybrid",
#             "maximum_diversity": "kmeans",
#             "fastest": "knn_cosine"
#         },
#         last_updated=datetime.now()
#     )

# @app.post("/feedback/", response_model=FeedbackOut)
# def submit_feedback(feedback: FeedbackIn):
#     """Submit user feedback on recommendations for system improvement."""
#     try:
#         # Map recipe names to IDs for sample data
#         name_to_id = {
#             "Oatmeal with Berries": 0,
#             "Grilled Chicken Salad": 1,
#             "Baked Salmon with Vegetables": 2
#         }

#         # Try to parse as int first, otherwise look up by name
#         try:
#             actual_recipe_id = int(feedback.recipe_id)
#         except ValueError:
#             actual_recipe_id = name_to_id.get(feedback.recipe_id, 0)  # Default to 0 if not found

#         logger.info(f"Received feedback from user {feedback.user_id} for recipe {feedback.recipe_id} (ID: {actual_recipe_id}): rating {feedback.rating}")

#         # In a real implementation, this would be stored in a database
#         # For now, we'll just log it and return a confirmation
#         feedback_id = f"fb_{feedback.user_id}_{actual_recipe_id}_{int(datetime.now().timestamp())}"

#         # Could implement feedback analysis here
#         # e.g., update user preferences, retrain models, etc.

#         return FeedbackOut(
#             status="success",
#             feedback_id=feedback_id,
#             message="Thank you for your feedback! It will help us improve our recommendations.",
#             timestamp=datetime.now()
#         )

#     except Exception as e:
#         logger.error(f"Error processing feedback: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")

# @app.post("/explain/", response_model=ExplanationResponse)
# def explain_recipe_recommendation_body(payload: ExplainIn):
#     """Get detailed explanation of why a recipe was recommended using a request body."""
#     try:
#         explanation = explain_recommendation(
#             recipe_id=payload.recipe_id,
#             target_vector=payload.nutrition_input,
#             bmi=payload.bmi,
#             goal=payload.goal
#         )

#         if "error" in explanation:
#             raise HTTPException(status_code=404, detail=explanation["error"])

#         return ExplanationResponse(
#             recipe_id=payload.recipe_id,
#             recipe_name=explanation.get("recipe_name", "Unknown Recipe"),
#             explanation=explanation,
#             model_used=explanation.get("model_used", "unknown"),
#             confidence=explanation.get("confidence", 0.0),
#             timestamp=datetime.now()
#         )

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error generating explanation: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")

# @app.get("/explain/{recipe_id}", response_model=ExplanationResponse)
# def explain_recipe_recommendation(
#     recipe_id: str,
#     nutrition_input: str = Query("2000,65,20,0,2000,300,25,50,150", description="Nutrition input as comma-separated values"),
#     bmi: Optional[float] = Query(None, gt=0, le=100, description="Body Mass Index"),
#     goal: Optional[str] = Query(None, pattern="^(weight_loss|muscle_gain|maintenance)$", description="Fitness goal"),
#     metric: str = Query("nutritional_mae", description="Selection metric")
# ):
#     """Get detailed explanation of why a recipe was recommended."""
#     try:
#         logger.info(f"Processing explanation request for recipe {recipe_id}")

#         # Map recipe names to IDs for sample data
#         name_to_id = {
#             "Oatmeal with Berries": 0,
#             "Grilled Chicken Salad": 1,
#             "Baked Salmon with Vegetables": 2
#         }

#         # Try to parse as int first, otherwise look up by name
#         try:
#             actual_id = int(recipe_id)
#         except ValueError:
#             actual_id = name_to_id.get(recipe_id, 0)  # Default to 0 if not found

#         # Sample explanation data for demo
#         sample_explanations = {
#             0: {
#                 "recipe_name": "Oatmeal with Berries",
#                 "model_used": "demo_sample",
#                 "confidence": 0.85,
#                 "explanation": {
#                     "nutritional_match": "High fiber content matches target requirements",
#                     "calorie_alignment": "320 calories fits within daily target range",
#                     "protein_content": "12g protein supports maintenance goals",
#                     "health_benefits": "Rich in antioxidants from berries",
#                     "similarity_score": 0.92,
#                     "ranking_factors": ["fiber_content", "calorie_density", "protein_quality"]
#                 }
#             },
#             1: {
#                 "recipe_name": "Grilled Chicken Salad",
#                 "model_used": "demo_sample",
#                 "confidence": 0.88,
#                 "explanation": {
#                     "nutritional_match": "Excellent protein-to-calorie ratio",
#                     "calorie_alignment": "380 calories appropriate for meal portion",
#                     "protein_content": "35g protein ideal for muscle maintenance",
#                     "health_benefits": "Low in processed carbs, high in nutrients",
#                     "similarity_score": 0.89,
#                     "ranking_factors": ["protein_content", "calorie_efficiency", "nutrient_density"]
#                 }
#             },
#             2: {
#                 "recipe_name": "Baked Salmon with Vegetables",
#                 "model_used": "demo_sample",
#                 "confidence": 0.90,
#                 "explanation": {
#                     "nutritional_match": "Omega-3 rich, supports heart health",
#                     "calorie_alignment": "420 calories suitable for dinner",
#                     "protein_content": "38g protein excellent for satiety",
#                     "health_benefits": "Anti-inflammatory properties from salmon",
#                     "similarity_score": 0.94,
#                     "ranking_factors": ["protein_quality", "fat_composition", "micronutrients"]
#                 }
#             }
#         }

#         explanation = sample_explanations.get(actual_id, {
#             "recipe_name": f"Sample Recipe {actual_id}",
#             "model_used": "demo_sample",
#             "confidence": 0.80,
#             "explanation": {
#                 "nutritional_match": "Balanced nutritional profile",
#                 "calorie_alignment": "Appropriate calorie content",
#                 "protein_content": "Adequate protein for goals",
#                 "health_benefits": "Supports overall health",
#                 "similarity_score": 0.85,
#                 "ranking_factors": ["balance", "completeness", "suitability"]
#             }
#         })

#         return ExplanationResponse(
#             recipe_id=actual_id,
#             recipe_name=explanation["recipe_name"],
#             explanation=explanation["explanation"],
#             model_used=explanation["model_used"],
#             confidence=explanation["confidence"],
#             timestamp=datetime.now()
#         )

#     except Exception as e:
#         logger.error(f"Error generating explanation: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(e)}")

# @app.get("/models/feature-importance", response_model=FeatureImportanceResponse)
# def get_feature_importance():
#     """Get feature importance analysis for nutrition-based recommendations."""
#     try:
#         importance_data = get_model_feature_importance()

#         if "error" in importance_data:
#             raise HTTPException(status_code=500, detail=importance_data["error"])

#         return FeatureImportanceResponse(
#             feature_importance=importance_data["feature_importance"],
#             methodology=importance_data["methodology"],
#             timestamp=datetime.now()
#         )

#     except Exception as e:
#         logger.error(f"Error getting feature importance: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Feature importance calculation failed: {str(e)}")

# @app.get("/analytics/recommendation-stats", response_model=StatisticsResponse)
# def get_recommendation_statistics(
#     nutrition_input: str = Query("2000,65,20,0,2000,300,25,50,150", description="Nutrition input as comma-separated values"),
#     metric: str = Query("nutritional_mae", pattern="^(nutritional_mae|diversity_score)$"),
#     bmi: Optional[float] = Query(None, gt=0, le=100),
#     goal: Optional[str] = Query(None, pattern="^(weight_loss|muscle_gain|maintenance)$")
# ):
#     """Get recommendation statistics and analytics."""
#     try:
#         logger.info(f"Processing analytics request with metric {metric}")

#         # Sample statistics for demo
#         sample_stats = {
#             "recommendation_count": 150,
#             "nutrition_statistics": {
#                 "average_calories": 385.5,
#                 "average_protein": 28.3,
#                 "average_carbs": 42.1,
#                 "average_fat": 16.7,
#                 "average_fiber": 6.2
#             },
#             "error_analysis": {
#                 "calories": {"mae": 45.2, "rmse": 62.1},
#                 "protein": {"mae": 3.8, "rmse": 5.2},
#                 "carbs": {"mae": 8.9, "rmse": 12.3},
#                 "fat": {"mae": 2.1, "rmse": 3.4}
#             },
#             "diversity_score": 0.78,
#             "model_used": "demo_sample",
#             "target_nutrition": {
#                 "calories": 2000.0,
#                 "fat": 65.0,
#                 "sat_fat": 20.0,
#                 "cholesterol": 0.0,
#                 "sodium": 2000.0,
#                 "carbs": 300.0,
#                 "fiber": 25.0,
#                 "sugar": 50.0,
#                 "protein": 150.0
#             }
#         }

#         return StatisticsResponse(
#             recommendation_count=sample_stats["recommendation_count"],
#             nutrition_statistics=sample_stats["nutrition_statistics"],
#             error_analysis=sample_stats["error_analysis"],
#             diversity_score=sample_stats["diversity_score"],
#             model_used=sample_stats["model_used"],
#             target_nutrition=sample_stats["target_nutrition"],
#             timestamp=datetime.now()
#         )

#     except Exception as e:
#         logger.error(f"Error generating statistics: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Statistics generation failed: {str(e)}")

# @app.post("/analytics/recommendation-stats", response_model=StatisticsResponse)
# def get_recommendation_statistics_body(payload: StatsRequest):
#     """Get detailed statistics about recommendation performance using a request body."""
#     try:
#         stats = get_recommendation_statistics_model(
#             nutrition_input=payload.nutrition_input,
#             metric=payload.metric,
#             bmi=payload.bmi,
#             goal=payload.goal
#         )

#         if "error" in stats:
#             raise HTTPException(status_code=500, detail=stats["error"])

#         return StatisticsResponse(
#             recommendation_count=stats["recommendation_count"],
#             nutrition_statistics=stats["nutrition_statistics"],
#             error_analysis=stats["error_analysis"],
#             diversity_score=stats["diversity_score"],
#             model_used=stats["model_used"],
#             target_nutrition=stats["target_nutrition"],
#             timestamp=datetime.now()
#         )

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error generating statistics: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Statistics generation failed: {str(e)}")
