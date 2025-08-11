import numpy as np
from tqdm import tqdm
# np.seterr(all='warn')
import pandas as pd
import ast

def jaccard_score(a: np.ndarray, b: np.ndarray) -> float:
    c = a + b
    intersection = np.zeros_like(c)
    intersection[c > 1] = 1
    union = np.zeros_like(c)
    union[c >= 1] = 1
    
    if np.sum(union) == 0:
        score = 0
    else:
        score = np.sum(intersection) / np.sum(union)

    return float(score)

def compute_item_similarity_matrix(inter: np.ndarray) -> np.ndarray:
    n_items = inter.shape[1]
    sim_matrix = np.zeros((n_items, n_items))
    for i in tqdm(range(n_items), desc="Computing item similarity matrix"):
        for j in range(i, n_items):
            score = jaccard_score(inter[:, i], inter[:, j])
            sim_matrix[i, j] = sim_matrix[j, i] = score
    return sim_matrix


def recTopK(inter_matr: np.ndarray, top_k: int, n: int, weights: dict = None, favorite_artists: list = [], favorite_genres: list = []) -> dict:
    n_users, n_items = inter_matr.shape
    sim_matrix = compute_item_similarity_matrix(inter_matr)

    recommendations = {}
    respective_scores = {}

    all_items = pd.read_csv('challenge_data/lfm-challenge.item', sep='\t')

    for user in tqdm(range(n_users), desc="Computing recommendations"):
        user_scores = np.zeros(n_items)

        user_interactions = inter_matr[user] == 1
        not_interacted_items = np.where(inter_matr[user] == 0)[0]

        for item in not_interacted_items:
            # Similarities between target item and items the user interacted with
            sim_scores = sim_matrix[item][user_interactions]
            top_sim_scores = np.sort(sim_scores)[-n:]  # take top-n
            model_scores = top_sim_scores.mean() if len(top_sim_scores) > 0 else 0.0
            final_score = model_scores

            if weights is not None:
                alpha = weights['alpha']
                beta = weights['beta']
                gamma = weights['gamma']

                user_favorite_artists = favorite_artists[user]
                user_favorite_genres = favorite_genres[user]

                item_genres = all_items[all_items['item_id'] == item]['genre']
                item_genres = ast.literal_eval(item_genres) if type(item_genres) == str else []
                item_artist = all_items[all_items['item_id'] == item]['artist'].values[0]

                genre_boost = 0
                artist_boost = 0

                for genre in item_genres:
                    if genre in user_favorite_genres:
                        rank = safe_index(user_favorite_genres, genre)+1
                        if rank != 0:
                            genre_boost += 1/float(np.log2(rank+1))

                if item_artist in user_favorite_artists:
                    rank = safe_index(user_favorite_artists, item_artist)+1
                    if rank != 0:
                        artist_boost = 1/float(np.log2(rank+1))

                final_score  = alpha * model_scores + beta * artist_boost + gamma * genre_boost


            user_scores[item] = final_score

        top_items = (-user_scores).argsort()[:top_k]
        recommendations[user] = top_items.tolist()
        respective_scores[user] = user_scores[top_items].tolist()

    return recommendations

def safe_index(arr, value):
    matches = np.where(arr == value)[0]
    return matches[0] if matches.size > 0 else -1