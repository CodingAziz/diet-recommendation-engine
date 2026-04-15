import numpy as np
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import pandas as pd
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

DATA_PATH = '../Data/dataset.csv'

NUTRITION_COLS = [
    'Calories', 'FatContent', 'SaturatedFatContent',
    'CholesterolContent', 'SodiumContent',
    'CarbohydrateContent', 'FiberContent',
    'SugarContent', 'ProteinContent'
]

# Load and preprocess data
df = pd.read_csv(DATA_PATH, compression='gzip')
df = df.dropna(subset=NUTRITION_COLS)
for col in NUTRITION_COLS:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=NUTRITION_COLS)
df['RecipeIngredientParts'] = df['RecipeIngredientParts'].apply(
    lambda x: x if isinstance(x, list) else str(x).split(';')
)
df.reset_index(drop=True, inplace=True)

# Scalers
minmax_scaler = MinMaxScaler()
X_minmax = minmax_scaler.fit_transform(df[NUTRITION_COLS])

std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(df[NUTRITION_COLS])

# Models
K_CANDIDATES = 50
TOP_K = 10

knn_cosine = NearestNeighbors(n_neighbors=K_CANDIDATES, metric='cosine')
knn_cosine.fit(X_minmax)

knn_euclidean = NearestNeighbors(n_neighbors=K_CANDIDATES, metric='euclidean')
knn_euclidean.fit(X_std)

N_CLUSTERS = 20
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_std)

N_COMPONENTS = 5
svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
X_svd = svd.fit_transform(X_std)

knn_svd = NearestNeighbors(n_neighbors=K_CANDIDATES, metric='cosine')
knn_svd.fit(X_svd)

def recommend_knn_cosine(target_vector, top_k=TOP_K):
    tv_scaled = minmax_scaler.transform([target_vector])
    _, indices = knn_cosine.kneighbors(tv_scaled)
    results = df.iloc[indices[0]].head(top_k).copy()
    return results

def recommend_knn_euclidean(target_vector, top_k=TOP_K):
    tv_scaled = std_scaler.transform([target_vector])
    _, indices = knn_euclidean.kneighbors(tv_scaled)
    results = df.iloc[indices[0]].head(top_k).copy()
    return results

def recommend_kmeans(target_vector, top_k=TOP_K):
    tv_scaled = std_scaler.transform([target_vector])
    centre_dists = euclidean_distances(tv_scaled, kmeans.cluster_centers_)[0]
    nearest_cluster = np.argmin(centre_dists)
    cluster_df = df[df['cluster'] == nearest_cluster].copy()
    cluster_scaled = std_scaler.transform(cluster_df[NUTRITION_COLS])
    sims = cosine_similarity(tv_scaled, cluster_scaled)[0]
    cluster_df['_sim'] = sims
    results = cluster_df.nlargest(top_k, '_sim').drop(columns='_sim')
    return results

def recommend_svd(target_vector, top_k=TOP_K):
    tv_scaled = std_scaler.transform([target_vector])
    tv_svd = svd.transform(tv_scaled)
    _, indices = knn_svd.kneighbors(tv_svd)
    results = df.iloc[indices[0]].head(top_k).copy()
    return results

def health_penalty(df_cands, bmi, goal):
    penalty = np.zeros(len(df_cands))
    if bmi >= 30:
        penalty += 0.01 * df_cands['Calories'].values
        penalty += 0.02 * df_cands['FatContent'].values
    if goal == 'weight_loss':
        penalty += 0.03 * df_cands['SugarContent'].values
    return penalty

def recommend_hybrid(target_vector, bmi, goal, top_k=TOP_K):
    tv_scaled = minmax_scaler.transform([target_vector])
    _, indices = knn_cosine.kneighbors(tv_scaled)
    candidates = df.iloc[indices[0]].copy()
    sims = cosine_similarity(
        [target_vector],
        minmax_scaler.transform(candidates[NUTRITION_COLS])
    )[0]
    candidates['similarity_score'] = sims
    candidates['health_penalty'] = health_penalty(candidates, bmi, goal)
    candidates['final_score'] = (
        0.7 * candidates['similarity_score']
        - 0.3 * candidates['health_penalty']
    )
    results = candidates.nlargest(top_k, 'final_score')
    return results

def recommend(dataframe, _input, ingredients=[], params={'n_neighbors':5,'return_distance':False}, metric='nutritional_mae', bmi=None, goal=None):
    target_vector = _input
    top_k = params['n_neighbors']
    if metric == 'nutritional_mae':
        if bmi is None:
            bmi = 25
        if goal is None:
            goal = 'maintenance'
        results = recommend_hybrid(target_vector, bmi, goal, top_k=top_k)
    elif metric == 'diversity_score':
        results = recommend_kmeans(target_vector, top_k=top_k)
    else:
        results = recommend_knn_cosine(target_vector, top_k=top_k)
    
    if ingredients:
        results = results[results['RecipeIngredientParts'].apply(lambda parts: any(ing.lower() in [p.lower() for p in parts] for ing in ingredients))]
        results = results.head(top_k)
    
    return results

def extract_quoted_strings(s):
    strings = re.findall(r'"([^"]*)"', s)
    return strings

def output_recommended_recipes(dataframe):
    if dataframe is not None:
        output=dataframe.copy()
        output=output.to_dict("records")
        for recipe in output:
            recipe['RecipeIngredientParts']=extract_quoted_strings(recipe['RecipeIngredientParts'])
            recipe['RecipeInstructions']=extract_quoted_strings(recipe['RecipeInstructions'])
    else:
        output=None
    return output

# Model Explainability Functions
def explain_recommendation(recipe_id: int, target_vector: list, bmi: float = None, goal: str = None):
    """
    Explain why a specific recipe was recommended using SHAP-like analysis

    Args:
        recipe_id: ID of the recipe to explain
        target_vector: Target nutritional values
        bmi: Body Mass Index for personalization
        goal: Fitness goal

    Returns:
        Dictionary containing explanation details
    """
    try:
        # Find the recipe in our dataset
        recipe_row = df[df.index == recipe_id]
        if recipe_row.empty:
            return {"error": "Recipe not found"}

        recipe_data = recipe_row.iloc[0]
        recipe_nutrition = recipe_data[NUTRITION_COLS].values

        # Calculate similarity scores
        similarity_scores = {}
        for i, nutrient in enumerate(NUTRITION_COLS):
            target_val = target_vector[i]
            recipe_val = recipe_nutrition[i]

            # Calculate how close the recipe value is to target (normalized)
            if target_val != 0:
                similarity = 1 - abs(recipe_val - target_val) / (abs(target_val) * 2)  # Allow 100% deviation
                similarity = max(0, min(1, similarity))  # Clamp to [0,1]
            else:
                similarity = 1.0 if recipe_val == 0 else 0.0

            similarity_scores[nutrient] = {
                'target': float(target_val),
                'recipe_value': float(recipe_val),
                'similarity': float(similarity),
                'difference': float(recipe_val - target_val)
            }

        # Calculate overall similarity
        overall_similarity = np.mean([s['similarity'] for s in similarity_scores.values()])

        # Calculate health penalties if using hybrid model
        health_penalty = 0.0
        penalty_factors = []

        if bmi is not None and goal is not None:
            if bmi >= 30:
                calorie_penalty = 0.01 * recipe_data['Calories']
                fat_penalty = 0.02 * recipe_data['FatContent']
                health_penalty += calorie_penalty + fat_penalty
                penalty_factors.extend([
                    {'factor': 'obesity_calorie_penalty', 'value': calorie_penalty},
                    {'factor': 'obesity_fat_penalty', 'value': fat_penalty}
                ])

            if goal == 'weight_loss':
                sugar_penalty = 0.03 * recipe_data['SugarContent']
                health_penalty += sugar_penalty
                penalty_factors.append({'factor': 'weight_loss_sugar_penalty', 'value': sugar_penalty})

        # Calculate final score (if hybrid)
        final_score = overall_similarity
        if health_penalty > 0:
            final_score = 0.7 * overall_similarity - 0.3 * health_penalty

        # Get top contributing factors
        sorted_factors = sorted(
            similarity_scores.items(),
            key=lambda x: x[1]['similarity'],
            reverse=True
        )

        explanation = {
            'recipe_id': int(recipe_id),
            'recipe_name': str(recipe_data.get('Name', 'Unknown Recipe')),
            'similarity_analysis': {
                'overall_similarity': float(overall_similarity),
                'top_contributing_factors': [
                    {
                        'nutrient': factor,
                        'target': data['target'],
                        'recipe_value': data['recipe_value'],
                        'similarity_score': data['similarity'],
                        'description': _generate_factor_description(factor, data)
                    } for factor, data in sorted_factors[:5]  # Top 5 factors
                ]
            },
            'health_considerations': penalty_factors if penalty_factors else None,
            'final_score': float(final_score),
            'model_used': 'hybrid' if bmi and goal else 'knn_cosine',
            'confidence': float(final_score)
        }

        return explanation

    except Exception as e:
        logger.error(f"Error explaining recommendation: {e}")
        return {"error": f"Explanation failed: {str(e)}"}

def _generate_factor_description(nutrient: str, data: dict) -> str:
    """Generate human-readable description for a similarity factor"""
    target = data['target']
    recipe_val = data['recipe_value']
    diff = data['difference']

    if abs(diff) < 0.1:  # Very close
        return f"Perfect match: {recipe_val:.1f} vs target {target:.1f}"
    elif abs(diff) / max(abs(target), 1) < 0.1:  # Close
        direction = "above" if diff > 0 else "below"
        return f"Very close: {recipe_val:.1f} vs target {target:.1f} ({abs(diff):.1f} {direction})"
    else:
        direction = "higher" if diff > 0 else "lower"
        percent_diff = abs(diff) / max(abs(target), 1) * 100
        return f"{direction.capitalize()} than target: {recipe_val:.1f} vs {target:.1f} ({percent_diff:.1f}% difference)"

def get_model_feature_importance() -> dict:
    """
    Calculate feature importance across all nutrition columns
    based on their variance and correlation with recommendations
    """
    try:
        # Calculate variance (how much recipes vary in each nutrient)
        variances = df[NUTRITION_COLS].var()

        # Calculate correlation with calories (as a proxy for overall nutritional content)
        correlations = df[NUTRITION_COLS].corr()['Calories'].abs()

        # Calculate information content (entropy-like measure)
        info_content = {}
        for col in NUTRITION_COLS:
            # Bin the values and calculate entropy
            binned = pd.cut(df[col], bins=10, labels=False)
            proportions = binned.value_counts(normalize=True)
            entropy = -sum(p * np.log(p + 1e-10) for p in proportions)
            info_content[col] = entropy

        # Combine metrics for overall importance
        importance_scores = {}
        for col in NUTRITION_COLS:
            # Weighted combination: 40% variance, 40% correlation, 20% information content
            score = (0.4 * variances[col] / variances.max() +
                    0.4 * correlations[col] +
                    0.2 * info_content[col] / max(info_content.values()))
            importance_scores[col] = float(score)

        # Sort by importance
        sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            'feature_importance': [
                {'nutrient': nutrient, 'importance_score': score}
                for nutrient, score in sorted_importance
            ],
            'methodology': 'Combined variance, correlation, and information content analysis'
        }

    except Exception as e:
        logger.error(f"Error calculating feature importance: {e}")
        return {"error": f"Feature importance calculation failed: {str(e)}"}

def get_recommendation_statistics(nutrition_input: list, metric: str = 'nutritional_mae',
                                bmi: float = None, goal: str = None) -> dict:
    """
    Get detailed statistics about recommendations for analysis
    """
    try:
        # Get recommendations
        recommendations = recommend(
            dataframe=None,
            _input=nutrition_input,
            ingredients=[],
            params={'n_neighbors': 10, 'return_distance': False},
            metric=metric,
            bmi=bmi,
            goal=goal
        )

        if recommendations is None or len(recommendations) == 0:
            return {"error": "No recommendations generated"}

        # Calculate statistics
        nutrition_stats = recommendations[NUTRITION_COLS].describe()

        # Calculate distances from target
        target_array = np.array(nutrition_input)
        distances = {}
        for col in NUTRITION_COLS:
            recipe_values = recommendations[col].values
            distances[col] = {
                'mean_absolute_error': float(np.mean(np.abs(recipe_values - target_array[NUTRITION_COLS.index(col)]))),
                'rmse': float(np.sqrt(np.mean((recipe_values - target_array[NUTRITION_COLS.index(col)]) ** 2))),
                'max_deviation': float(np.max(np.abs(recipe_values - target_array[NUTRITION_COLS.index(col)])))
            }

        # Diversity analysis (cosine distance between recommendations)
        if len(recommendations) > 1:
            scaled_recs = minmax_scaler.transform(recommendations[NUTRITION_COLS])
            diversity_matrix = cosine_similarity(scaled_recs)
            # Average distance from each recipe to others
            diversity_scores = []
            for i in range(len(diversity_matrix)):
                distances_from_others = [diversity_matrix[i][j] for j in range(len(diversity_matrix)) if i != j]
                diversity_scores.append(np.mean(distances_from_others))

            avg_diversity = float(np.mean(diversity_scores))
        else:
            avg_diversity = 0.0

        return {
            'recommendation_count': len(recommendations),
            'nutrition_statistics': nutrition_stats.to_dict(),
            'error_analysis': distances,
            'diversity_score': avg_diversity,
            'model_used': 'hybrid' if bmi and goal else 'knn_cosine',
            'target_nutrition': dict(zip(NUTRITION_COLS, nutrition_input))
        }

    except Exception as e:
        logger.error(f"Error generating recommendation statistics: {e}")
        return {"error": f"Statistics generation failed: {str(e)}"}
