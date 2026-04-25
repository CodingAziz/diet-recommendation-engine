from ..models.model_selector import get_model_for_metric
from ..utils.data_loader import load_dataset
from ..utils.preprocessing import parse_ingredients
from ..models.recommender import Recommender


class RecommendationService:
    def __init__(self):
        df = parse_ingredients(load_dataset())
        self.recommender = Recommender(df)

    def predict(
        self,
        input_vec,
        metric,
        bmi,
        goal,
        top_k,
        ingredients=None
    ):
        model = get_model_for_metric(metric)

        results = self.recommender.recommend(
            input_vec=input_vec,
            model=model,
            bmi=bmi,
            goal=goal,
            top_k=top_k
        )

        return results, model