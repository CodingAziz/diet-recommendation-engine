# import numpy as np
# import re
# from pathlib import Path
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.neighbors import NearestNeighbors
# from sklearn.cluster import KMeans
# from sklearn.decomposition import TruncatedSVD
# from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
# import pandas as pd
# import warnings
# import logging

# warnings.filterwarnings('ignore')
# logger = logging.getLogger(__name__)

# NUTRITION_COLS = [
#     'Calories', 'FatContent', 'SaturatedFatContent',
#     'CholesterolContent', 'SodiumContent',
#     'CarbohydrateContent', 'FiberContent',
#     'SugarContent', 'ProteinContent'
# ]

# DATA_PATH='../Data/dataset_uncompressed.csv'

 
# df = pd.read_csv(DATA_PATH) # read csv file

# df[NUTRITION_COLS] = df[NUTRITION_COLS].apply(pd.to_numeric, errors='coerce') # vectorized conversion to numeric form
# df = df.dropna(subset=NUTRITION_COLS) # drop any null numeric values

# df['RecipeIngredientParts'] = df['RecipeIngredientParts'].apply(
#     lambda x: x if isinstance(x, list)
#     else [] if pd.isna(x)
#     else str(x).split(';')
# ) # splits the ingredients into parts

# df.reset_index(drop=True, inplace=True) # reset index of the dataset


# df = pd.DataFrame() # create an empty dataframe

# minmax_scaler = MinMaxScaler()
# X_minmax = minmax_scaler.fit_transform(df[NUTRITION_COLS]) # Scales each feature to range of [0-1]

# std_scaler = StandardScaler()
# X_std = std_scaler.fit_transform(df[NUTRITION_COLS]) # Transforms data to mean or std_dev

# # KNN
# K_CANDIDATES = 50
# TOP_K = 10

# knn_cosine = NearestNeighbors(n_neighbors=K_CANDIDATES, metric='cosine')
# knn_cosine.fit(X_minmax)

# knn_euclidean = NearestNeighbors(n_neighbors=K_CANDIDATES, metric='euclidean')
# knn_euclidean.fit(X_std)

# # KMeans
# N_CLUSTERS = 20
# kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
# df['cluster'] = kmeans.fit_predict(X_std)

# # SVD
# N_COMPONENTS = 5
# svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
# X_svd = svd.fit_transform(X_std)

# # KNN_SVD
# knn_svd = NearestNeighbors(n_neighbors=K_CANDIDATES, metric='cosine')
# knn_svd.fit(X_svd)

# # Model functions
# def recommend_knn_cosine(target_vector, top_k=10):
#     # Return empty dataframe for demo
#     return pd.DataFrame()

# def recommend_knn_euclidean(target_vector, top_k=10):
#     return pd.DataFrame()

# def recommend_kmeans(target_vector, top_k=10):
#     return pd.DataFrame()

# def recommend_svd(target_vector, top_k=10):
#     return pd.DataFrame()

# def health_penalty(df_cands, bmi, goal):
#     return np.zeros(len(df_cands)) if len(df_cands) > 0 else np.array([])

# def recommend_hybrid(target_vector, bmi, goal, top_k=10):
#     return pd.DataFrame()

# def recommend(dataframe, _input, ingredients=[], params={'n_neighbors':5,'return_distance':False}, metric='nutritional_mae', bmi=None, goal=None):
#     # Return empty dataframe for demo (main.py handles sample data)
#     return pd.DataFrame()

# def extract_quoted_strings(s):
#     strings = re.findall(r'"([^"]*)"', s)
#     return strings

# def output_recommended_recipes(dataframe):
#     # Return None for demo (main.py handles sample data)
#     return None

# def explain_recommendation(recipe_id: int, target_vector: list, bmi: float = None, goal: str = None):
#     # Stub implementation for demo
#     return {
#         'recipe_id': recipe_id,
#         'recipe_name': f'Sample Recipe {recipe_id}',
#         'model_used': 'demo_sample',
#         'confidence': 0.8,
#         'explanation': {
#             'nutritional_match': 'Balanced nutritional profile',
#             'similarity_score': 0.85
#         }
#     }

# def _generate_factor_description(nutrient: str, data: dict) -> str:
#     """Generate human-readable description for a similarity factor"""
#     target = data['target']
#     recipe_val = data['recipe_value']
#     diff = data['difference']

#     if abs(diff) < 0.1:  # Very close
#         return f"Perfect match: {recipe_val:.1f} vs target {target:.1f}"
#     elif abs(diff) / max(abs(target), 1) < 0.1:  # Close
#         direction = "above" if diff > 0 else "below"
#         return f"Very close: {recipe_val:.1f} vs target {target:.1f} ({abs(diff):.1f} {direction})"
#     else:
#         direction = "higher" if diff > 0 else "lower"
#         percent_diff = abs(diff) / max(abs(target), 1) * 100
#         return f"{direction.capitalize()} than target: {recipe_val:.1f} vs {target:.1f} ({percent_diff:.1f}% difference)"

# def get_model_feature_importance() -> dict:
#     """
#     Calculate feature importance across all nutrition columns
#     based on their variance and correlation with recommendations
#     """
#     try:
#         # Calculate variance (how much recipes vary in each nutrient)
#         variances = df[NUTRITION_COLS].var()

#         # Calculate correlation with calories (as a proxy for overall nutritional content)
#         correlations = df[NUTRITION_COLS].corr()['Calories'].abs()

#         # Calculate information content (entropy-like measure)
#         info_content = {}
#         for col in NUTRITION_COLS:
#             # Bin the values and calculate entropy
#             binned = pd.cut(df[col], bins=10, labels=False)
#             proportions = binned.value_counts(normalize=True)
#             entropy = -sum(p * np.log(p + 1e-10) for p in proportions)
#             info_content[col] = entropy

#         # Combine metrics for overall importance
#         importance_scores = {}
#         for col in NUTRITION_COLS:
#             # Weighted combination: 40% variance, 40% correlation, 20% information content
#             score = (0.4 * variances[col] / variances.max() +
#                     0.4 * correlations[col] +
#                     0.2 * info_content[col] / max(info_content.values()))
#             importance_scores[col] = float(score)

#         # Sort by importance
#         sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)

#         return {
#             'feature_importance': [
#                 {'nutrient': nutrient, 'importance_score': score}
#                 for nutrient, score in sorted_importance
#             ],
#             'methodology': 'Combined variance, correlation, and information content analysis'
#         }

#     except Exception as e:
#         logger.error(f"Error calculating feature importance: {e}")
#         return {"error": f"Feature importance calculation failed: {str(e)}"}

# def get_recommendation_statistics(nutrition_input: list, metric: str = 'nutritional_mae',
#                                 bmi: float = None, goal: str = None) -> dict:
#     """
#     Get detailed statistics about recommendations for analysis
#     """
#     try:
#         # Get recommendations
#         recommendations = recommend(
#             dataframe=None,
#             _input=nutrition_input,
#             ingredients=[],
#             params={'n_neighbors': 10, 'return_distance': False},
#             metric=metric,
#             bmi=bmi,
#             goal=goal
#         )

#         if recommendations is None or len(recommendations) == 0:
#             return {"error": "No recommendations generated"}

#         # Calculate statistics
#         nutrition_stats = recommendations[NUTRITION_COLS].describe()

#         # Calculate distances from target
#         target_array = np.array(nutrition_input)
#         distances = {}
#         for col in NUTRITION_COLS:
#             recipe_values = recommendations[col].values
#             distances[col] = {
#                 'mean_absolute_error': float(np.mean(np.abs(recipe_values - target_array[NUTRITION_COLS.index(col)]))),
#                 'rmse': float(np.sqrt(np.mean((recipe_values - target_array[NUTRITION_COLS.index(col)]) ** 2))),
#                 'max_deviation': float(np.max(np.abs(recipe_values - target_array[NUTRITION_COLS.index(col)])))
#             }

#         # Diversity analysis (cosine distance between recommendations)
#         if len(recommendations) > 1:
#             scaled_recs = minmax_scaler.transform(recommendations[NUTRITION_COLS])
#             diversity_matrix = cosine_similarity(scaled_recs)
#             # Average distance from each recipe to others
#             diversity_scores = []
#             for i in range(len(diversity_matrix)):
#                 distances_from_others = [diversity_matrix[i][j] for j in range(len(diversity_matrix)) if i != j]
#                 diversity_scores.append(np.mean(distances_from_others))

#             avg_diversity = float(np.mean(diversity_scores))
#         else:
#             avg_diversity = 0.0

#         return {
#             'recommendation_count': len(recommendations),
#             'nutrition_statistics': nutrition_stats.to_dict(),
#             'error_analysis': distances,
#             'diversity_score': avg_diversity,
#             'model_used': 'hybrid' if bmi and goal else 'knn_cosine',
#             'target_nutrition': dict(zip(NUTRITION_COLS, nutrition_input))
#         }

#     except Exception as e:
#         logger.error(f"Error generating recommendation statistics: {e}")
#         return {"error": f"Statistics generation failed: {str(e)}"}
