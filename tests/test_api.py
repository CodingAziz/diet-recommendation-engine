import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "FastAPI_Backend"))

try:
    from fastapi.testclient import TestClient
    from main import app
    
    client = TestClient(app)
    api_available = True
except ImportError:
    api_available = False


@pytest.mark.skipif(not api_available, reason="FastAPI not available in test environment")
class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert "health_check" in response.json()
    
    def test_predict_endpoint_valid_input(self):
        """Test /predict/ endpoint with valid input"""
        payload = {
            "nutrition_input": [1500.0, 50.0, 15.0, 100.0, 500.0, 150.0, 20.0, 25.0, 60.0],
            "ingredients": [],
            "params": {"n_neighbors": 5, "return_distance": False},
            "bmi": 24.5,
            "goal": "weight_loss",
            "metric": "nutritional_mae"
        }
        
        response = client.post("/predict/", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "output" in data
    
    def test_predict_endpoint_invalid_metric(self):
        """Test /predict/ endpoint with invalid metric"""
        payload = {
            "nutrition_input": [1500.0, 50.0, 15.0, 100.0, 500.0, 150.0, 20.0, 25.0, 60.0],
            "ingredients": [],
            "params": {"n_neighbors": 5, "return_distance": False},
            "bmi": 24.5,
            "goal": "weight_loss",
            "metric": "invalid_metric"
        }
        
        response = client.post("/predict/", json=payload)
        # Should still work but use default model
        assert response.status_code == 200
    
    def test_predict_endpoint_missing_fields(self):
        """Test /predict/ endpoint with missing required fields"""
        payload = {
            "nutrition_input": [1500.0, 50.0, 15.0, 100.0, 500.0, 150.0, 20.0, 25.0, 60.0],
        }
        
        response = client.post("/predict/", json=payload)
        # Should fail due to missing required fields
        assert response.status_code == 422
    
    def test_predict_endpoint_invalid_nutrition_array_size(self):
        """Test /predict/ with wrong nutrition array size"""
        payload = {
            "nutrition_input": [1500.0, 50.0],  # Only 2 values, need 9
            "ingredients": [],
            "params": {"n_neighbors": 5, "return_distance": False},
            "bmi": 24.5,
            "goal": "weight_loss",
            "metric": "nutritional_mae"
        }
        
        response = client.post("/predict/", json=payload)
        assert response.status_code == 422


@pytest.mark.skipif(not api_available, reason="FastAPI not available in test environment")
class TestAPIDataValidation:
    """Test API data validation"""
    
    def test_valid_bmi_range(self):
        """Test valid BMI range"""
        payload = {
            "nutrition_input": [1500.0, 50.0, 15.0, 100.0, 500.0, 150.0, 20.0, 25.0, 60.0],
            "ingredients": [],
            "params": {"n_neighbors": 5, "return_distance": False},
            "bmi": 18.5,  # Valid BMI
            "goal": "weight_loss",
            "metric": "nutritional_mae"
        }
        
        response = client.post("/predict/", json=payload)
        assert response.status_code == 200
    
    def test_valid_goals(self):
        """Test valid fitness goals"""
        goals = ["weight_loss", "muscle_gain", "maintenance"]
        
        for goal in goals:
            payload = {
                "nutrition_input": [1500.0, 50.0, 15.0, 100.0, 500.0, 150.0, 20.0, 25.0, 60.0],
                "ingredients": [],
                "params": {"n_neighbors": 5, "return_distance": False},
                "bmi": 24.5,
                "goal": goal,
                "metric": "nutritional_mae"
            }
            
            response = client.post("/predict/", json=payload)
            assert response.status_code == 200
    
    def test_response_structure(self):
        """Test that response has correct structure"""
        payload = {
            "nutrition_input": [1500.0, 50.0, 15.0, 100.0, 500.0, 150.0, 20.0, 25.0, 60.0],
            "ingredients": [],
            "params": {"n_neighbors": 5, "return_distance": False},
            "bmi": 24.5,
            "goal": "weight_loss",
            "metric": "nutritional_mae"
        }
        
        response = client.post("/predict/", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "output" in data
        # output can be None or a list of recipes
        if data["output"] is not None:
            assert isinstance(data["output"], list)
