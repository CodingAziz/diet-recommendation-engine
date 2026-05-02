from ..models.model_selector import get_model_for_metric
from ..utils.data_loader import load_dataset
from ..utils.preprocessing import parse_ingredients
from ..models.recommender import Recommender
from ..utils.formatters import format_recipe_dataframe, clean_ingredients, clean_instructions

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

        if ingredients:
            def match(recipe_ingredients):
                recipe_ingredients = [e.lower() for e in recipe_ingredients]

                return any(
                    any(i.lower() in ri for ri in recipe_ingredients)
                    for i in ingredients
                )

            results = results[
                results["RecipeIngredientParts"].apply(match)
            ]
        
        results = format_recipe_dataframe(results)
        results["RecipeIngredientParts"] = results["RecipeIngredientParts"].apply(clean_ingredients)
        results["RecipeInstructions"] = results["RecipeInstructions"].apply(clean_instructions)

        return results, model