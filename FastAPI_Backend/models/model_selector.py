def get_model_for_metric(metric: str) -> str:
    if metric == "nutritional_mae":
        return "knn_euclidean"
    elif metric == "cosine":
        return "knn_cosine"
    else:
        return "hybrid"
