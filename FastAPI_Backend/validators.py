"""Input validation for Diet Recommendation System"""
from typing import List, Tuple
from config import settings


class NutritionValidator:
    """Validator for nutrition inputs"""
    
    @staticmethod
    def validate_nutrition_input(nutrition: List[float]) -> Tuple[bool, str]:
        """
        Validate nutrition input array
        
        Args:
            nutrition: List of 9 nutrition values
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not nutrition or len(nutrition) != 9:
            return False, f"Expected 9 nutrition values, got {len(nutrition) if nutrition else 0}"
        
        calories, fat, sat_fat, chol, sodium, carbs, fiber, sugar, protein = nutrition
        
        # Validate calories
        if not settings.MIN_CALORIES <= calories <= settings.MAX_CALORIES:
            return False, f"Calories must be between {settings.MIN_CALORIES} and {settings.MAX_CALORIES}"
        
        # Validate fat
        if not settings.MIN_MACRO_VALUE <= fat <= settings.MAX_FAT:
            return False, f"Fat content must be between {settings.MIN_MACRO_VALUE} and {settings.MAX_FAT}"
        
        # Validate saturated fat
        if not settings.MIN_MACRO_VALUE <= sat_fat <= fat:
            return False, f"Saturated fat must be between 0 and total fat content"
        
        # Validate cholesterol
        if not settings.MIN_MACRO_VALUE <= chol <= 1000:
            return False, "Cholesterol content out of valid range"
        
        # Validate sodium
        if not settings.MIN_MACRO_VALUE <= sodium <= settings.MAX_SODIUM:
            return False, f"Sodium must be between {settings.MIN_MACRO_VALUE} and {settings.MAX_SODIUM}"
        
        # Validate carbs
        if not settings.MIN_MACRO_VALUE <= carbs <= 500:
            return False, "Carbohydrate content out of valid range"
        
        # Validate fiber
        if not settings.MIN_MACRO_VALUE <= fiber <= carbs:
            return False, "Fiber must be between 0 and carbohydrate content"
        
        # Validate sugar
        if not settings.MIN_MACRO_VALUE <= sugar <= carbs:
            return False, "Sugar must be between 0 and carbohydrate content"
        
        # Validate protein
        if not settings.MIN_MACRO_VALUE <= protein <= 300:
            return False, "Protein content out of valid range"
        
        return True, ""
    
    @staticmethod
    def validate_bmi(bmi: float) -> Tuple[bool, str]:
        """
        Validate BMI value
        
        Args:
            bmi: BMI value
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(bmi, (int, float)):
            return False, "BMI must be a number"
        
        if bmi < 10 or bmi > 60:
            return False, "BMI must be between 10 and 60"
        
        return True, ""
    
    @staticmethod
    def validate_goal(goal: str) -> Tuple[bool, str]:
        """
        Validate fitness goal
        
        Args:
            goal: Fitness goal string
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if goal not in settings.VALID_GOALS:
            return False, f"Goal must be one of {settings.VALID_GOALS}, got '{goal}'"
        
        return True, ""
    
    @staticmethod
    def validate_metric(metric: str) -> Tuple[bool, str]:
        """
        Validate metric selection
        
        Args:
            metric: Metric string
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if metric not in settings.VALID_METRICS:
            return False, f"Metric must be one of {settings.VALID_METRICS}, got '{metric}'"
        
        return True, ""
    
    @staticmethod
    def validate_n_neighbors(n_neighbors: int) -> Tuple[bool, str]:
        """
        Validate number of neighbors
        
        Args:
            n_neighbors: Number of neighbors
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(n_neighbors, int):
            return False, "n_neighbors must be an integer"
        
        if n_neighbors < 1 or n_neighbors > settings.MAX_N_NEIGHBORS:
            return False, f"n_neighbors must be between 1 and {settings.MAX_N_NEIGHBORS}"
        
        return True, ""
    
    @staticmethod
    def validate_all(nutrition: List[float], bmi: float, goal: str, metric: str, n_neighbors: int) -> Tuple[bool, str]:
        """
        Validate all inputs together
        
        Args:
            nutrition: Nutrition values
            bmi: BMI value
            goal: Fitness goal
            metric: Metric selection
            n_neighbors: Number of neighbors
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        validators = [
            (NutritionValidator.validate_nutrition_input(nutrition), "Nutrition"),
            (NutritionValidator.validate_bmi(bmi), "BMI"),
            (NutritionValidator.validate_goal(goal), "Goal"),
            (NutritionValidator.validate_metric(metric), "Metric"),
            (NutritionValidator.validate_n_neighbors(n_neighbors), "N-Neighbors"),
        ]
        
        for (is_valid, error), field_name in validators:
            if not is_valid:
                return False, f"{field_name} validation failed: {error}"
        
        return True, ""
