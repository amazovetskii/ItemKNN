import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as cosine_similarity

def compute_itemknn_scores(seen_item_ids: list, item_embeddings: pd.DataFrame, k: int = 10) -> np.ndarray:
    similarity_matrix = cosine_similarity(item_embeddings)

    recommendation_scores = np.zeros(len(item_embeddings))
    for idx in seen_item_ids:
        sim_vector = similarity_matrix[idx]
        top_indices = np.argsort(sim_vector)[::-1][1:k+1]
        recommendation_scores[top_indices] += sim_vector[top_indices]

    recommendation_scores /= len(seen_item_ids)
    for idx in seen_item_ids:
        recommendation_scores[idx] = -np.inf

    return recommendation_scores

def cb_itemknn_recommendation(seen_item_ids: list, item_embeddings: pd.DataFrame, top_k: int=10, knn_k: int=10) -> np.ndarray:
    item_embeddings = item_embeddings.sort_values('item_id').reset_index(drop=True)
    item_ids = item_embeddings['item_id'].values
    embeddings = item_embeddings.drop(columns=['item_id']).values

    scores = compute_itemknn_scores(seen_item_ids, embeddings, knn_k)

    recommended_indices = np.argsort(scores)[::-1][:top_k]
    recommended_item_ids = item_ids[recommended_indices]

    return recommended_item_ids
