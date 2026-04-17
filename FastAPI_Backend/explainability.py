"""Model explainability and interpretability"""
from typing import Dict, List, Tuple
import numpy as np


class ModelExplainer:
    """Explains model predictions"""
    
    @staticmethod
    def explain_similarity_scores(
        nutrition_vector: List[float],
        recipe_nutrition: List[float],
        nutrition_cols: List[str]
    ) -> Dict[str, float]:
        """
        Explain contribution of each nutrient to similarity score
        
        Args:
            nutrition_vector: User's target nutrition
            recipe_nutrition: Recipe's nutrition values
            nutrition_cols: Nutrition column names
        
        Returns:
            Dictionary of nutrient contributions
        """
        contributions = {}
        total_diff = 0.0
        
        for i, col in enumerate(nutrition_cols):
            diff = abs(nutrition_vector[i] - recipe_nutrition[i])
            total_diff += diff
            contributions[col] = diff
        
        # Normalize contributions
        if total_diff > 0:
            for col in contributions:
                contributions[col] = (contributions[col] / total_diff) * 100
        
        return contributions
    
    @staticmethod
    def explain_ranking(
        recipe: Dict,
        similarity_score: float,
        health_penalty: float = 0.0,
        bmi: float = None,
        goal: str = None
    ) -> Dict:
        """
        Explain why a recipe was ranked
        
        Args:
            recipe: Recipe data
            similarity_score: Cosine similarity score
            health_penalty: Health penalty factor
            bmi: User's BMI
            goal: User's fitness goal
        
        Returns:
            Explanation dictionary
        """
        explanation = {
            "recipe_name": recipe.get("Name", "Unknown"),
            "similarity_score": round(similarity_score, 4),
            "health_penalty": round(health_penalty, 4),
            "final_score": round(0.7 * similarity_score - 0.3 * health_penalty, 4),
            "reasoning": []
        }
        
        # Add reasoning
        if similarity_score > 0.7:
            explanation["reasoning"].append("High nutritional match with target profile")
        
        if health_penalty > 0:
            if bmi and bmi >= 30:
                explanation["reasoning"].append("Health adjustments applied for overweight status")
            if goal == "weight_loss":
                explanation["reasoning"].append("Sugar content penalized for weight loss goal")
        
        explanation["reasoning"].append(f"Fitness goal: {goal}")
        
        return explanation
    
    @staticmethod
    def get_feature_importance(
        model_type: str,
        nutrition_cols: List[str]
    ) -> Dict[str, float]:
        """
        Get feature importance based on model type
        
        Args:
            model_type: Type of model (hybrid, kmeans, knn_cosine, etc.)
            nutrition_cols: List of nutrition columns
        
        Returns:
            Dictionary of feature importances
        """
        importances = {}
        
        if model_type == "hybrid":
            # For hybrid, protein and calories are most important
            base_importance = 1.0 / len(nutrition_cols)
            for col in nutrition_cols:
                if col in ["ProteinContent", "Calories"]:
                    importances[col] = base_importance * 1.5
                elif col in ["SugarContent", "SodiumContent"]:
                    importances[col] = base_importance * 1.2
                else:
                    importances[col] = base_importance
        
        elif model_type == "kmeans":
            # For K-Means, all features have equal importance
            importance = 1.0 / len(nutrition_cols)
            for col in nutrition_cols:
                importances[col] = importance
        
        else:  # KNN variants
            # For KNN, approximate importances
            importance = 1.0 / len(nutrition_cols)
            for col in nutrition_cols:
                importances[col] = importance
        
        # Normalize
        total = sum(importances.values())
        for col in importances:
            importances[col] = importances[col] / total
        
        return importances
    
    @staticmethod
    def get_model_characteristics(model_type: str) -> Dict:
        """
        Get characteristics of each model
        
        Args:
            model_type: Type of model
        
        Returns:
            Dictionary of model characteristics
        """
        characteristics = {
            "hybrid": {
                "name": "Hybrid Scoring (KNN + Health Penalty)",
                "strengths": [
                    "Personalizes recommendations based on health metrics (BMI, goal)",
                    "Balances nutritional accuracy with health considerations",
                    "Fast inference time"
                ],
                "weaknesses": [
                    "May favor similar recipes",
                    "Less diverse recommendations"
                ],
                "best_for": "Users with specific health goals and constraints",
                "algorithm": "Cosine similarity KNN with health-based penalty"
            },
            "kmeans": {
                "name": "K-Means Clustering",
                "strengths": [
                    "Highly diverse recommendations",
                    "Explores wider recipe space",
                    "Good for discovering new recipes"
                ],
                "weaknesses": [
                    "May have lower nutritional accuracy",
                    "Depends on cluster quality"
                ],
                "best_for": "Users who want diverse recipe suggestions",
                "algorithm": "K-Means clustering with inter-cluster ranking"
            },
            "knn_cosine": {
                "name": "KNN with Cosine Similarity",
                "strengths": [
                    "Scale-invariant metric",
                    "Fast inference",
                    "Good baseline"
                ],
                "weaknesses": [
                    "Ignores absolute magnitudes",
                    "May miss important differences"
                ],
                "best_for": "General-purpose recommendations",
                "algorithm": "K-Nearest Neighbors with cosine similarity"
            },
            "knn_euclidean": {
                "name": "KNN with Euclidean Distance",
                "strengths": [
                    "Considers absolute differences",
                    "Penalizes large deviations",
                    "Intuitive distance metric"
                ],
                "weaknesses": [
                    "Scale-dependent",
                    "May need normalization"
                ],
                "best_for": "When absolute nutritional differences matter",
                "algorithm": "K-Nearest Neighbors with Euclidean distance"
            },
            "svd": {
                "name": "SVD Collaborative Filtering",
                "strengths": [
                    "Captures latent nutritional patterns",
                    "Good generalization",
                    "Reduces dimensionality"
                ],
                "weaknesses": [
                    "More complex model",
                    "Requires more training data"
                ],
                "best_for": "Large datasets with complex patterns",
                "algorithm": "Truncated SVD with KNN in latent space"
            }
        }
        
        return characteristics.get(model_type, {"name": "Unknown", "strengths": [], "weaknesses": []})
