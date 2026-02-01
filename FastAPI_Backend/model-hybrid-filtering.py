import pickle
import json
import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity

# =========================================================
# Load artifacts ONCE (at import time)
# =========================================================

with open("scaler.pkl", "rb") as f:
    SCALER = pickle.load(f)

with open("knn.pkl", "rb") as f:
    KNN_MODEL = pickle.load(f)

with open("nutrition_schema.json", "r") as f:
    NUTRITION_COLS = json.load(f)

# =========================================================
# Utility functions
# =========================================================

def extract_quoted_strings(s):
    if pd.isna(s):
        return []
    return re.findall(r'"([^"]*)"', s)


def filter_by_ingredients(df, ingredients):
    if not ingredients:
        return df

    regex = "".join(f"(?=.*{ing})" for ing in ingredients)
    return df[df["RecipeIngredientParts"].str.contains(
        regex, case=False, regex=True, na=False
    )]


def apply_goal_rules(df, goals=None):
    """
    goals example:
    {
        "max_calories": 500,
        "low_sugar": True,
        "high_protein": True
    }
    """
    if not goals:
        return df

    if "max_calories" in goals:
        df = df[df["Calories"] <= goals["max_calories"]]

    if goals.get("low_sugar"):
        df = df[df["SugarContent"] <= 10]

    if goals.get("high_protein"):
        df = df[df["ProteinContent"] >= 15]

    return df


# =========================================================
# Core recommendation logic
# =========================================================

def recommend(
    dataset: pd.DataFrame,
    nutrition_input: list,
    ingredients: list = None,
    params: dict = None,
    goals: dict = None
):
    """
    End-to-end recommendation pipeline:
    1. Ingredient filtering
    2. Goal-based rule filtering
    3. KNN similarity (PKL)
    4. Fallback cosine similarity
    """

    ingredients = ingredients or []
    params = params or {"n_neighbors": 5}

    # -----------------------------
    # Step 1: Ingredient filtering
    # -----------------------------
    filtered_df = filter_by_ingredients(dataset, ingredients)

    # -----------------------------
    # Step 2: Goal-based rules
    # -----------------------------
    filtered_df = apply_goal_rules(filtered_df, goals)

    if filtered_df.empty:
        return None

    # -----------------------------
    # Step 3: Prepare nutrition vectors
    # -----------------------------
    X = filtered_df[NUTRITION_COLS].values
    X_scaled = SCALER.transform(X)

    user_vec = np.array(nutrition_input).reshape(1, -1)
    user_vec_scaled = SCALER.transform(user_vec)

    n_neighbors = params.get("n_neighbors", 5)

    # -----------------------------
    # Step 4: KNN similarity (PRIMARY)
    # -----------------------------
    try:
        distances, indices = KNN_MODEL.kneighbors(
            user_vec_scaled,
            n_neighbors=min(n_neighbors, len(X_scaled))
        )

        recommended_df = filtered_df.iloc[indices[0]].copy()
        recommended_df["score"] = 1 - distances[0]  # similarity score

    # -----------------------------
    # Step 5: Fallback similarity
    # -----------------------------
    except Exception:
        similarity = cosine_similarity(user_vec_scaled, X_scaled)[0]
        filtered_df["score"] = similarity
        recommended_df = filtered_df.sort_values(
            "score", ascending=False
        ).head(n_neighbors)

    return recommended_df


# =========================================================
# Output formatting (API-safe)
# =========================================================

def output_recommended_recipes(df: pd.DataFrame):
    if df is None or df.empty:
        return None

    output = df.copy()

    output["RecipeIngredientParts"] = output["RecipeIngredientParts"].apply(
        extract_quoted_strings
    )

    output["RecipeInstructions"] = output["RecipeInstructions"].apply(
        extract_quoted_strings
    )

    return output.drop(columns=["score"], errors="ignore").to_dict("records")
