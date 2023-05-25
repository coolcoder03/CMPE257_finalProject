import os
import pandas as pd
import joblib
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict, Counter
import math
import numpy as np
import random
import copy
import gc
from gensim.models import Word2Vec
from sklearn.neighbors import NearestNeighbors

def load_datasets_and_mappings():
    """Load training data and id-to-type mappings."""
    training_data = pd.read_parquet('../input/otto-full-optimized-memory-footprint/train.parquet')
    id_to_type_mapping = joblib.load('../input/otto-full-optimized-memory-footprint/id2type.pkl')
    type_to_id_mapping = joblib.load('../input/otto-full-optimized-memory-footprint/type2id.pkl')

    return training_data, id_to_type_mapping, type_to_id_mapping


def preprocess_training_data(training_data, config):
    """Preprocess the training data."""
    training_data['aid'] = training_data['aid'].astype('int32').astype('str')

    # Randomly sample sessions for training
    sampled_sessions = random.sample(list(training_data['session'].unique()), config['train_session_num'])
    training_data = training_data.query('session in @sampled_sessions').reset_index(drop=True)

    training_data['time_stamp'] = pd.to_datetime(training_data['ts'], unit='s').dt.strftime('%Y-%m-%d')

    return training_data


def generate_word2vec_embeddings(data):
    """Generate Word2Vec embeddings for session sequences."""
    session_sequences = data.groupby('session')['aid'].apply(list).tolist()

    # Train Word2Vec model
    model = Word2Vec(session_sequences, min_count=1, sg=1)
    word_vectors = model.wv

    return word_vectors


def recommend_items(session_items, word_vectors, nearest_neighbors, popular_items):
    """Recommend items based on the given session items using Word2Vec and nearest neighbors."""
    item_embeddings = []
    for item in session_items:
        if item in word_vectors:
            item_embeddings.append(word_vectors[item])

    if len(item_embeddings) > 0:
        session_embedding = np.mean(item_embeddings, axis=0)
        _, indices = nearest_neighbors.kneighbors([session_embedding])
        similar_items = nearest_neighbors._fit_X[indices.flatten()]
        recommended_items = [item for item in similar_items[0] if item not in session_items]
        recommended_items = recommended_items[:20]  # Limit to 20 recommendations
    else:
        recommended_items = []

    if len(recommended_items) < 20:
        return recommended_items + popular_items[:20 - len(recommended_items)]
    else:
        return recommended_items


def load_and_preprocess_test_data():
    """Load and preprocess test data."""
    test_data = pd.read_parquet('../input/otto-full-optimized-memory-footprint/test.parquet')
    test_data['aid'] = test_data['aid'].astype('int32').astype('str')
    test_data['time_stamp'] = pd.to_datetime(test_data['ts'], unit='s').dt.strftime('%Y-%m-%d')
    test_data = test_data.sort_values(["session", "type", "ts"])
    session_to_item_ids = test_data.groupby('session')['aid'].agg(list).to_dict()

    return session_to_item_ids


def generate_recommendations(session_to_item_ids, word_vectors, nearest_neighbors, popular_items):
    """Generate item recommendations for each session."""
    session_ids = []
    recommended_item_lists = []
    for session_id, session_items in tqdm(session_to_item_ids.items()):
        recommended_items = recommend_items(session_items, word_vectors, nearest_neighbors, popular_items)
        session_ids.append(session_id)
        recommended_item_lists.append(recommended_items)

    return session_ids, recommended_item_lists


def create_submission_file(session_ids, recommended_item_lists, id_to_type_mapping):
    """Create a submission file with the recommended items for each session type."""
    submission_df = pd.DataFrame()
    submission_df['session_type'] = session_ids
    submission_df['labels'] = [' '.join([str(item) for item in item_list]) for item_list in recommended_item_lists]

    submission_list = []
    for type_ in [0, 1, 2]:
        type_specific_df = submission_df.copy()
        type_specific_df['session_type'] = type_specific_df['session_type'].apply(lambda x: f'{x}_{id_to_type_mapping[type_]}')
        submission_list.append(type_specific_df)
    submission_df = pd.concat(submission_list, axis=0)

    submission_df.to_csv('submission.csv', index=False)


def main():
    config = {'train_session_num': 12899779}
    training_data, id_to_type_mapping, _ = load_datasets_and_mappings()
    training_data = preprocess_training_data(training_data, config)
    word_vectors = generate_word2vec_embeddings(training_data)
    session_sequences = training_data.groupby('session')['aid'].apply(list).tolist()
    nearest_neighbors = NearestNeighbors(metric='cosine')
    nearest_neighbors.fit(word_vectors[session_sequences])
    del training_data, session_sequences
    gc.collect()

    session_to_item_ids = load_and_preprocess_test_data()
    popular_items = list(training_data['aid'].value_counts().index)
    session_ids, recommended_item_lists = generate_recommendations(session_to_item_ids, word_vectors, nearest_neighbors, popular_items)
    create_submission_file(session_ids, recommended_item_lists, id_to_type_mapping)


if __name__ == "__main__":
    main()
