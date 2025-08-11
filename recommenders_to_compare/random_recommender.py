import numpy as np

def random_recommender(user_ids, val_inter_matrix, top_k):
    np.random.seed(42)
    n_users, n_items = val_inter_matrix.shape
    recommendations = np.zeros((n_users, top_k), dtype=np.int32)
    for user in range(n_users):
        seen_items = np.where(val_inter_matrix[user] == 1)[0]
        unseen_items = np.setdiff1d(np.arange(n_items), seen_items)
        selected_items = np.random.choice(unseen_items, size=top_k, replace=False)
        recommendations[user] = selected_items
        
    return recommendations