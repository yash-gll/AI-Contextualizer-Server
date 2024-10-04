from flask import Flask, request, jsonify
from scipy.spatial.distance import cosine
from flask_cors import CORS, cross_origin
import os
import numpy as np
import pandas as pd
import torch
import json
import io 

from dotenv import load_dotenv
from io import StringIO, BytesIO
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import DBSCAN
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestCentroid
from sklearn.metrics.pairwise import cosine_similarity

from model import initialize_model, intialize_tokenizer, clean_text, tokenize_and_encode_text, generate_embeddings, match_experiences
import firebase_admin
from firebase_admin import credentials, firestore, storage
import nltk
nltk.download('punkt_tab')

# Load environment variables
load_dotenv()

# Firebase initialization
# Get the JSON string from environment variables
firebase_config_str = os.getenv('FIREBASE_CONFIG')
if firebase_config_str is None:
    raise ValueError("FIREBASE_CONFIG environment variable is not set.")

firebase_config = json.loads(firebase_config_str)

cred = credentials.Certificate(firebase_config)
firebase_admin.initialize_app(cred)
db = firestore.client()
bucket = storage.bucket(os.getenv('FIREBASE_BUCKET'))

USERNAME = os.getenv('USERNAME')

app = Flask(__name__)
CORS(app)

try:
    model = initialize_model('sentence-transformers/all-MiniLM-L6-v2')
    tokenizer = intialize_tokenizer('sentence-transformers/all-MiniLM-L6-v2')
except Exception as e:
    app.logger.error(f"Failed to load model: {e}")
    model, tokenizer = None, None



def save_embeddings(embeddings, file_path):
    buffer = io.BytesIO()
    np.save(buffer, embeddings.numpy())
    buffer.seek(0)
    blob = bucket.blob(file_path)
    blob.upload_from_file(buffer, content_type='application/octet-stream')
    print(f"Embeddings saved to {file_path}.")

def load_embeddings(file_path):
    blob = bucket.blob(file_path)
    buffer = io.BytesIO(blob.download_as_bytes())
    return torch.tensor(np.load(buffer, allow_pickle=True))

def file_exists_in_firebase(filename):
    """Check if a file exists in Firebase Storage."""
    blob = bucket.blob(filename)
    return blob.exists()

# Firebase Storage utility functions
def save_csv_to_firebase(df, file_path):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    blob = bucket.blob(file_path)
    blob.upload_from_string(csv_buffer.getvalue(), content_type='text/csv')
    print(f"Saved {file_path} to Firebase Storage.")

def load_csv_from_firebase(file_path):
    blob = bucket.blob(file_path)
    csv_string = blob.download_as_text()
    return pd.read_csv(StringIO(csv_string))


def fetch_collection_data(collection_name):
    collection_ref = db.collection(collection_name)
    docs = collection_ref.stream()

    data_list = []
    for doc in docs:
        data_dict = doc.to_dict()
        data_dict['id'] = doc.id  # Store document ID as well
        data_list.append(data_dict)

    return data_list

def clean_and_prepare_data(data_list):
    # Example of cleaning and preparing data
    prepared_data = []
    for data in data_list:
        cleaned_data = {
            'id': data['id'],
            'title': data['experience_title'].strip(),
            'description': data['experience_description'].replace('\n', ' ').strip(),
            'student_attendance': data['student_data']['student_attendance'].strip(),
            'student_class_participation': data['student_data']['student_class_participation'].strip(),
            'student_gender': data['student_data']['student_gender'].strip(),
            'student_learning_disability': data['student_data']['student_learning_disability'].strip(),
            'student_overall_performance': data['student_data']['student_overall_performance'].strip(),
        }
        prepared_data.append(cleaned_data)

    return prepared_data



def process_experiences(file_path, model, tokenizer):
    try:
        df = load_csv_from_firebase(file_path)
        reviews = df['description'].dropna().tolist()
        clean_reviews = [clean_text(review) for review in reviews]
        encoded_reviews_data, reviews_token_embeddings = tokenize_and_encode_text(model, tokenizer, clean_reviews)
        reviews_embeddings = generate_embeddings(reviews_token_embeddings, encoded_reviews_data)
        return reviews, reviews_embeddings
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None



def get_clusters(data):
    dbscan = DBSCAN(eps=1.4, min_samples=6)
    clusters = dbscan.fit_predict(data)
    return clusters



def process_lesson_plan_text(text, model, tokenizer, pca):
    try:
        clean_lesson_plan = clean_text(text)
        encoded_lesson_plan, lesson_plan_token_embeddings = tokenize_and_encode_text(model, tokenizer, clean_lesson_plan)
        lesson_plan_embeddings = generate_embeddings(lesson_plan_token_embeddings, encoded_lesson_plan)
        lesson_plan_embedding_np = lesson_plan_embeddings.detach().numpy()
        reduced_lesson_plan_embedding = pca.transform(lesson_plan_embedding_np)
        return clean_lesson_plan, reduced_lesson_plan_embedding
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None
 


def get_top_recommendations(text_data, combined_data, cluster_num, lesson_text_embedding, top_n=5):
    filtered_experiences = text_data[text_data['cluster'] == cluster_num]
    filtered_embeddings = combined_data[combined_data['cluster'] == cluster_num]

    # Ensure we only deal with the embedding columns
    embedding_columns = [col for col in filtered_embeddings.columns if 'dim_' in col]
    similarity_scores = cosine_similarity(filtered_embeddings[embedding_columns].to_numpy(), lesson_text_embedding)

    # Normalize the similarity scores
    normalized_similarities = (similarity_scores.flatten() - similarity_scores.min()) / (similarity_scores.max() - similarity_scores.min())

    # Get indices of top N scores
    top_indices = np.argsort(-normalized_similarities)[:top_n]
    top_recommendations = filtered_experiences.iloc[top_indices]['id'].tolist()

    return top_recommendations

def fetch_and_process_data():
    # Fetch data from Firebase Firestore, clean and process it, then save to Firebase Storage
    data_list = fetch_collection_data('Experiences')
    cleaned_data = clean_and_prepare_data(data_list)
    df = pd.DataFrame(cleaned_data)
    experiences_csv_path = f'{USERNAME}/experiences.csv'
    save_csv_to_firebase(df, experiences_csv_path)
    return experiences_csv_path

def perform_data_combination_and_reduction(experiences_df, embeddings):
    # Additional data processing and feature engineering
    student_data = pd.DataFrame({
        'class_participation': experiences_df['student_class_participation'],
        'overall_performance': experiences_df['student_overall_performance']
    })

    onehot_encoder = OneHotEncoder()
    student_encoded = onehot_encoder.fit_transform(student_data).toarray()
    student_encoded_df = pd.DataFrame(student_encoded, columns=onehot_encoder.get_feature_names_out())
    combined_data = np.hstack((embeddings.numpy(), student_encoded))

    teacher_preferences = {
            'class_participation_importance': 0.9,
            'overall_performance_importance': 0.7
        }
    
    for i, column_name in enumerate(student_data.columns):
        if 'class_participation' in column_name:
            student_encoded[:, i] *= teacher_preferences['class_participation_importance']
        elif 'overall_performance' in column_name:
            student_encoded[:, i] *= teacher_preferences['overall_performance_importance']

    # Dimensionality reduction using UMAP
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=15, metric='euclidean')

    # Fit UMAP and transform the data
    reduced_data = reducer.fit_transform(embeddings)
    reduced_df = pd.DataFrame(reduced_data, columns=[f'dim_{i}' for i in range(reduced_data.shape[1])])
    student_encoded_df = pd.DataFrame(student_encoded, columns=onehot_encoder.get_feature_names_out(student_data.columns))
    combined_data = pd.concat([reduced_df, student_encoded_df], axis=1)
    return combined_data



@app.route('/api/get-recommendations', methods=['POST'])
def get_recommendations():
    data = request.json

    experiences_filename = f'{USERNAME}/experiences.csv'
    experiences_embeddings_filename = f'{USERNAME}/experiences_embeddings.npy'
    experiences_cluster_filename = f'{USERNAME}/experiences_cluster.csv'
    combined_filename = f'{USERNAME}/combined_data.csv'


    if not file_exists_in_firebase(experiences_filename):
        fetch_and_process_data()
        print(f"Saved experiences to Firebase")

    experiences_df = load_csv_from_firebase(experiences_filename)

    if not file_exists_in_firebase(experiences_embeddings_filename):
        _, experiences_embeddings_miniLM = process_experiences(experiences_filename, model, tokenizer)
        save_embeddings(experiences_embeddings_miniLM, experiences_embeddings_filename)

    experiences_embeddings_miniLM = load_embeddings(experiences_embeddings_filename)

    if not file_exists_in_firebase(combined_filename):
        combined_data_df = perform_data_combination_and_reduction(experiences_df, experiences_embeddings_miniLM)
        save_csv_to_firebase(combined_data_df, combined_filename)
        print(f"Combined data saved to Firebase Storage as {combined_filename}")
    else:
        print(f"{combined_filename} already exists.")
        combined_data_df = load_csv_from_firebase(combined_filename)


    clusters = get_clusters(combined_data_df)
    combined_data_df['cluster'] = clusters
    
    if not file_exists_in_firebase(experiences_cluster_filename):
        blob = bucket.blob(experiences_filename)
        blob.download_to_filename(experiences_filename)
        text_data = pd.read_csv(experiences_filename)
        os.remove(experiences_filename)
        
        text_data['cluster'] = clusters
        save_csv_to_firebase(text_data, experiences_cluster_filename)
        print(f"Saved {experiences_cluster_filename} experiences with cluster info to Firebase Storage")
    else:
        print(f"{experiences_cluster_filename} already exists.")
        text_data = load_csv_from_firebase(experiences_cluster_filename)


    pca = PCA(n_components=15)
    pca.fit(experiences_embeddings_miniLM)

    teacher_preferences = {
        "class_participation_High": 0.8,
        "class_participation_Low": 0,
        "class_participation_Mid": 0,
        "overall_performance_A": 0.7,
        "overall_performance_B": 0,
        "overall_performance_C": 0,
        "overall_performance_F": 0,
    }
    preference_vector = np.array([value for value in teacher_preferences.values()])

    results = {}
    response = {}
    for key, value in data.items():
        if key in ['Grade', 'Duration', 'Subject']:
            continue
        else:
            text = value['content']
            _, reduced_text_embedding = process_lesson_plan_text(text, model, tokenizer, pca)
            full_feature_vector = np.concatenate((reduced_text_embedding, preference_vector.reshape(1, -1)), axis=1)

            clf = NearestCentroid()
            filtered_embeddings = combined_data_df[combined_data_df['cluster'] != -1]
            clf.fit(filtered_embeddings.drop('cluster', axis=1), filtered_embeddings['cluster'])
            closest_cluster = clf.predict(full_feature_vector.reshape(1, -1))
            
            results[key] = {'Embeddings': full_feature_vector, 'Cluster': int(closest_cluster[0])}

            top_experience_ids = get_top_recommendations(text_data, combined_data_df, int(closest_cluster[0]), reduced_text_embedding)
            response[key] = top_experience_ids

    return jsonify(response), 200

if __name__ == '__main__':
    app.run(debug=True)