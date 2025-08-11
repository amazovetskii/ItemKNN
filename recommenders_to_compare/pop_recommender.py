import numpy as np

def pop_recommender(
        user_ids: list,
        inter_matrix: np.ndarray,
        top_k: int) -> np.array:
    recommendations = []
    for user in user_ids:
        inter_sum_by_item = np.sum(inter_matrix, 0)
        sorted_items = np.argsort(inter_sum_by_item)
        inters_of_user = np.nonzero(inter_matrix[user])
        inter_sum_by_item[inters_of_user] = 0
        sorted_items = (-inter_sum_by_item).argsort()
        top_pop = sorted_items[:top_k]
        recommendations.append(top_pop)

    return recommendations