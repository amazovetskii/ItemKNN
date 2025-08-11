import numpy as np

def svd_decompose(inter_matr: np.ndarray, f: int = 50) -> (np.ndarray, np.ndarray):
    U_final = None
    V_final = None
    U, s, Vh = np.linalg.svd(inter_matr, full_matrices=False)
    U_final = U[:, :f] @ np.diag(s[:f] ** 0.5)  # users x features
    V_final = (np.diag(s[:f] ** 0.5) @ Vh[:f, :]).T  # items x features

    return U_final, V_final


def svd_recommend_to_list(user_id: int, seen_item_ids: list, U: np.ndarray, V: np.ndarray, topK: int):
    recs = None

    scores = U @ V.T
    u_scores = scores[user_id]
    u_scores[seen_item_ids] = -np.inf
    m = min(topK, scores.shape[1])
    recs = (-u_scores).argsort()[:m]

    return np.array(recs), u_scores[recs]