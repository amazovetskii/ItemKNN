import pandas as pd
import numpy as np
from evaluate_algorithm import inter_matr_implicit, main
import subprocess

from item_knn import recTopK
from recommenders_to_compare.random_recommender import random_recommender
from recommenders_to_compare.pop_recommender import pop_recommender
from recommenders_to_compare.svd_recommender import svd_decompose, svd_recommend_to_list
from recommenders_to_compare.cb_item_knn import cb_itemknn_recommendation

top_k = 10

# Reading data
def read(dataset, file):
    return pd.read_csv('challenge_data/' + dataset + '.' + file, sep='\t')

def write_recommendations(recommendations):
    with open('recommendations.tsv', 'w') as f:
        for user_id, item_ids in zip(inters_test['user_id'].unique(), recommendations):
            line = f"{user_id}\t{','.join(item_ids.astype(str))}\n"
            f.write(line)

items = read("lfm-challenge", 'item')
users = read("lfm-challenge", 'user')
item_embeddings = read("lfm-challenge", 'musicnn')
train_inters = read("lfm-challenge", 'inter_train')
inters_test = read("lfm-challenge", 'inter_test')

# Split train_inters into train and validation sets
train_inter_matrix = inter_matr_implicit(users=train_inters['user_id'].max()+1, items=train_inters['item_id'].max()+1, interactions=train_inters)
test_inter_matrix = inter_matr_implicit(users=inters_test['user_id'].max()+1, items=inters_test['item_id'].max()+1, interactions=inters_test)

recommendations = []
recommendations_unsorted = recTopK(train_inter_matrix, top_k=top_k, n=2)
for user_id in inters_test['user_id'].unique():
    recommendations.append(recommendations_unsorted[user_id])
recommendations = np.array(recommendations)

write_recommendations(recommendations)

result = subprocess.run(["python3", "evaluate_algorithm.py", "--submission", "./recommendations.tsv", "--target", "./challenge_data/lfm-challenge.inter_test"],
            capture_output=True,
            text=True)
accuracy = float(result.stdout.split("\n")[1].split(" ")[-1][:6])
print(accuracy)