import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import torch
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
    
def initialize_model(model_name):
    try:
        model = AutoModel.from_pretrained(model_name)
        return model
    except Exception as e:
        print(f"Error initializing model: {e}")
        return None
    

def intialize_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
    except Exception as e:
        print(f"Error initializing tokenizer: {e}")
        return None
    

def clean_text(text):
    if not text:
        return ""
    
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    word_tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in word_tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    cleaned_text = ' '.join(lemmatized_tokens)

    return cleaned_text


def tokenize_and_encode_text(model, tokenizer, data):
	# Tokenize lesson plan
	encoded_data = tokenizer(data, padding=True, truncation=True, return_tensors='pt')
	# Compute token embeddings for lesson plan
	with torch.no_grad():
		token_embeddings = model(**encoded_data)

	return encoded_data, token_embeddings


# Perform mean pooling
def mean_pooling(model_output, attention_mask):
	token_embeddings = model_output[0]
	input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
	return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def generate_embeddings(token_embeddings, encoded_data):
	intial_embedding = mean_pooling(token_embeddings, encoded_data['attention_mask'])
	final_embedding = F.normalize(intial_embedding, p=2, dim=1)
	return final_embedding


def match_experiences(lesson_plan_embedding, experiences_embeddings, filepath, threshold):
    # Calculate cosine similarity and filter relevant reviews
    try:
        df = pd.read_csv(filepath)
        if 'description' not in df.columns or 'id' not in df.columns:
            raise ValueError("CSV does not contain necessary columns ('id' and 'description')")
        
        if lesson_plan_embedding is not None and experiences_embeddings is not None:
            similarities = cosine_similarity(lesson_plan_embedding, experiences_embeddings)[0]

            relevant_experience_ids = [df.iloc[i]['id'] for i in range(len(similarities)) if similarities[i] > threshold]

            print("Relevant experience IDs:", relevant_experience_ids)
            return relevant_experience_ids
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
