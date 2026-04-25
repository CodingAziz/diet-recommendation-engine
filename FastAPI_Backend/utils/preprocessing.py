import pandas as pd
import re

NUTRITION_COLS = [
    "Calories","FatContent","SaturatedFatContent",
    "CholesterolContent","SodiumContent",
    "CarbohydrateContent","FiberContent",
    "SugarContent","ProteinContent"
]

def parse_ingredients(df):
    def clean(x):
        if x is None:
            return []
        
        if isinstance(x, float) and pd.isna(x):
            return []
        
        if isinstance(x, str):
            return [i.strip() for i in x.split(",")]
        
        if isinstance(x, list):
            return x
        
        return []

    df["RecipeIngredientParts"] = df["RecipeIngredientParts"].apply(clean)
    return df

def preprocess(df):
    # convert nutrition to numeric
    df[NUTRITION_COLS] = df[NUTRITION_COLS].apply(pd.to_numeric, errors="coerce")

    # drop invalid rows
    df = df.dropna(subset=NUTRITION_COLS)

    # Parse ingredients
    df["RecipeIngredientParts"] = df["RecipeIngredientParts"].apply(parse_ingredients)

    # reset index of the new df after dropping null values
    df.reset_index(drop=True, inplace=True)

    return df
