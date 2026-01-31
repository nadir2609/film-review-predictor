import numpy as np
import pandas as pd

import os

import re

import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

train_data = pd.read_csv('data/raw/train_data.csv')
test_data = pd.read_csv('data/raw/test_data.csv')


def lowercase_text(text):
    return text.lower()

def remove_html_tags(text):
    text = re.sub(r'<.*?>', '', text)
    return text

def remove_urls(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    return text

def remove_special_characters(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def remove_extra_whitespace(text):
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords_tokenize_lemmatize(text):
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

def preprocess_text(text):
    text = lowercase_text(text)
    text = remove_html_tags(text)
    text = remove_urls(text)
    text = remove_special_characters(text)
    text = remove_extra_whitespace(text)
    text = remove_stopwords_tokenize_lemmatize(text)
    
    return text

train_data['review'] = train_data['review'].apply(preprocess_text)
test_data['review'] = test_data['review'].apply(preprocess_text)

data_path = os.path.join("data","processed")

os.makedirs(data_path, exist_ok=True)

train_data.to_csv(os.path.join(data_path,"train_processed.csv"), index=False)
test_data.to_csv(os.path.join(data_path,"test_processed.csv"), index=False)
