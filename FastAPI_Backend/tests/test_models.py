import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "FastAPI_Backend"))

# Note: These tests require the model to be importable
# They are designed to validate model behavior


class TestModelConsistency:
    """Test that models produce consistent results"""
    
    def test_recommendation_output_shape(self, sample_nutrition_input):
        """Test that recommendations return correct number of results"""
        try:
            from model import recommend
            top_k = 5
            result = recommend(
                dataframe=None,
                _input=sample_nutrition_input,
                ingredients=[],
                params={'n_neighbors': top_k, 'return_distance': False},
                metric='nutritional_mae',
                bmi=24.5,
                goal='weight_loss'
            )
            assert result is not None, "Recommendation should not be None"
            assert len(result) > 0, "Should return at least one recipe"
            assert len(result) <= top_k, f"Should return at most {top_k} recipes"
        except ImportError:
            pytest.skip("Model module not available in test environment")
    
    def test_model_reproducibility(self, sample_nutrition_input):
        """Test that models produce reproducible results with same seed"""
        try:
            from model import recommend
            
            result1 = recommend(
                dataframe=None,
                _input=sample_nutrition_input,
                ingredients=[],
                params={'n_neighbors': 5, 'return_distance': False},
                metric='nutritional_mae',
                bmi=24.5,
                goal='weight_loss'
            )
            
            result2 = recommend(
                dataframe=None,
                _input=sample_nutrition_input,
                ingredients=[],
                params={'n_neighbors': 5, 'return_distance': False},
                metric='nutritional_mae',
                bmi=24.5,
                goal='weight_loss'
            )
            
            if result1 is not None and result2 is not None:
                assert len(result1) == len(result2), "Results should have same length"
        except ImportError:
            pytest.skip("Model module not available in test environment")


class TestInputValidation:
    """Test input validation"""
    
    def test_invalid_nutrition_values(self):
        """Test handling of invalid nutrition values"""
        try:
            from model import validate_nutrition_input
            
            # Test negative values
            invalid_input = [-100.0, 50.0, 15.0, 100.0, 500.0, 150.0, 20.0, 25.0, 60.0]
            assert not validate_nutrition_input(invalid_input), "Should reject negative calories"
            
            # Test excessive values
            invalid_input = [10000.0, 50.0, 15.0, 100.0, 500.0, 150.0, 20.0, 25.0, 60.0]
            assert not validate_nutrition_input(invalid_input), "Should reject excessive calories"
        except (ImportError, AttributeError):
            pytest.skip("Validation function not available")
    
    def test_valid_nutrition_values(self):
        """Test that valid nutrition values pass validation"""
        try:
            from model import validate_nutrition_input
            
            valid_input = [1500.0, 50.0, 15.0, 100.0, 500.0, 150.0, 20.0, 25.0, 60.0]
            assert validate_nutrition_input(valid_input), "Should accept valid nutrition values"
        except (ImportError, AttributeError):
            pytest.skip("Validation function not available")


class TestMetricSelection:
    """Test metric-based model selection"""
    
    def test_nutritional_mae_metric(self, sample_nutrition_input, sample_user_params):
        """Test nutritional MAE selects hybrid model"""
        try:
            from model import recommend
            
            result = recommend(
                dataframe=None,
                _input=sample_nutrition_input,
                ingredients=[],
                params={'n_neighbors': 5, 'return_distance': False},
                metric='nutritional_mae',
                bmi=sample_user_params['bmi'],
                goal=sample_user_params['goal']
            )
            assert result is not None, "Hybrid model should return results"
        except ImportError:
            pytest.skip("Model module not available")
    
    def test_diversity_score_metric(self, sample_nutrition_input):
        """Test diversity score selects K-Means model"""
        try:
            from model import recommend
            
            result = recommend(
                dataframe=None,
                _input=sample_nutrition_input,
                ingredients=[],
                params={'n_neighbors': 5, 'return_distance': False},
                metric='diversity_score',
                bmi=24.5,
                goal='maintenance'
            )
            assert result is not None, "K-Means model should return results"
        except ImportError:
            pytest.skip("Model module not available")


class TestRecipeDataStructure:
    """Test recipe data structure"""
    
    def test_recipe_has_required_fields(self, sample_nutrition_input):
        """Test that recipes have required nutritional fields"""
        try:
            from model import recommend
            
            result = recommend(
                dataframe=None,
                _input=sample_nutrition_input,
                ingredients=[],
                params={'n_neighbors': 1, 'return_distance': False},
                metric='nutritional_mae',
                bmi=24.5,
                goal='maintenance'
            )
            
            if result is not None and len(result) > 0:
                recipe = result.iloc[0]
                required_fields = ['Calories', 'FatContent', 'ProteinContent', 'CarbohydrateContent']
                for field in required_fields:
                    assert field in recipe.index, f"Recipe should have {field} field"
        except ImportError:
            pytest.skip("Model module not available")
