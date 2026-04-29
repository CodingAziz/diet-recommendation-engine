import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

NUTRITION_COLS = [
    "Calories","FatContent","SaturatedFatContent",
    "CholesterolContent","SodiumContent",
    "CarbohydrateContent","FiberContent",
    "SugarContent","ProteinContent"
]

class Recommender:

    def __init__(self, df): # Initialized variables
        self.df = df.reset_index(drop=True)

        # scaling
        self.minmax_scaler = MinMaxScaler()
        self.std_scaler = StandardScaler()

        self.X_minmax = self.minmax_scaler.fit_transform(df[NUTRITION_COLS])
        self.X_std = self.std_scaler.fit_transform(df[NUTRITION_COLS])

        # KNN cosine
        self.knn_cosine = NearestNeighbors(n_neighbors=50, metric="cosine")
        self.knn_cosine.fit(self.X_minmax)

        # KNN euclidean
        self.knn_euclidean = NearestNeighbors(n_neighbors=50, metric='euclidean')
        self.knn_euclidean.fit(self.X_std)

        # Kmeans
        self.kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
        self.df['cluster'] = self.kmeans.fit_predict(self.X_std)

        # SVD
        self.svd = TruncatedSVD(n_components=5, random_state=42)
        self.X_svd = self.svd.fit_transform(self.X_std)

        self.knn_svd = NearestNeighbors(n_neighbors=50, metric='cosine')
        self.knn_svd.fit(self.X_svd)

    def _prepare_input(self, input_vec):
        # Case 1: frontend sends dict (BEST)
        if isinstance(input_vec, dict):
            df = pd.DataFrame([input_vec])

        # Case 2: frontend sends list/array
        else:
            df = pd.DataFrame([input_vec], columns=NUTRITION_COLS)

        # Ensure correct column order
        df = df[NUTRITION_COLS]

        return df

    # KNN cosine
    def recommend_knn_cosine(self, input_vec, top_k=10):
        df_input = self._prepare_input(input_vec)
        x = self.minmax_scaler.transform(df_input)
        dist, idx = self.knn_cosine.kneighbors(x)
        return self._build_result(idx[0], dist[0], top_k)

    # KNN euclidean
    def recommend_knn_euclidean(self, input_vec, top_k=10):
        df_input = self._prepare_input(input_vec)
        x = self.std_scaler.transform(df_input)
        dist, idx = self.knn_euclidean.kneighbors(x)
        return self._build_result(idx[0], dist[0], top_k)

    # Kmeans
    def recommend_kmeans(self, input_vec, top_k=10):
        df_input = self._prepare_input(input_vec)
        x = self.std_scaler.transform(df_input)
        cluster = self.kmeans.predict(x)[0]

        cluster_df = self.df[self.df['cluster'] == cluster]

        # fallback if cluster too small
        if len(cluster_df) < top_k:
            return cluster_df.head(top_k)

        return cluster_df.sample(top_k)

    # SVD + KNN
    def recommend_svd(self, input_vec, top_k=10):
        df_input = self._prepare_input(input_vec)
        x = self.std_scaler.transform(df_input)
        x_svd = self.svd.transform(x)

        dist, idx = self.knn_svd.kneighbors(x_svd)
        return self._build_result(idx[0], dist[0], top_k)

    # health penalty 
    def health_penalty(self, df_cands, bmi, goal):
        if df_cands.empty:
            return np.array([])

        penalties = []

        for _, row in df_cands.iterrows():
            penalty = 0

            if goal == "weight_loss":
                penalty += row["Calories"] * 0.01
            elif goal == "muscle_gain":
                penalty -= row["ProteinContent"] * 0.01

            if bmi and bmi > 25:
                penalty += row["FatContent"] * 0.01

            penalties.append(penalty)

        return np.array(penalties)

    # hybrid - combination of all three
    def recommend_hybrid(self, input_vec, bmi=None, goal=None, top_k=10):

        df1 = self.recommend_knn_cosine(input_vec, 30)
        df2 = self.recommend_knn_euclidean(input_vec, 30)
        df3 = self.recommend_svd(input_vec, 30)

        combined = pd.concat([df1, df2, df3])

        # Only dedupe on numeric columns (safe)
        combined = combined.drop_duplicates(subset=NUTRITION_COLS)

        # Apply health penalty
        penalties = self.health_penalty(combined, bmi, goal)

        if len(penalties) > 0:
            if len(penalties) == len(combined):
                combined["final_score"] -= penalties
            combined = combined.sort_values(by="final_score", ascending=False)

        return combined.head(top_k)

    # recommend functions
    def recommend(self, input_vec, model="hybrid", bmi=None, goal=None, top_k=10):

        if model == "knn_cosine":
            return self.recommend_knn_cosine(input_vec, top_k)

        elif model == "knn_euclidean":
            return self.recommend_knn_euclidean(input_vec, top_k)

        elif model == "kmeans":
            return self.recommend_kmeans(input_vec, top_k)

        elif model == "svd":
            return self.recommend_svd(input_vec, top_k)

        elif model == "hybrid":
            return self.recommend_hybrid(input_vec, bmi, goal, top_k)

        else:
            raise ValueError("Invalid model type")

    # helper function to build results
    def _build_result(self, indices, distances, top_k):
        df = self.df.iloc[indices].copy()
        df["score"] = 1 - distances
        return df.head(top_k)