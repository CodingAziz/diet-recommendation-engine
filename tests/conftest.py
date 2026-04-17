import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_nutrition_input():
    """Sample nutrition input for testing"""
    return [1500.0, 50.0, 15.0, 100.0, 500.0, 150.0, 20.0, 25.0, 60.0]


@pytest.fixture
def sample_user_params():
    """Sample user parameters"""
    return {
        "bmi": 24.5,
        "goal": "weight_loss",
        "metric": "nutritional_mae"
    }


@pytest.fixture
def sample_ingredients():
    """Sample ingredients list"""
    return ["chicken", "rice", "broccoli"]
