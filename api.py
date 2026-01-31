"""
FastAPI Application for IMDB Sentiment Analysis
Provides API endpoint for sentiment prediction
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

# Initialize FastAPI app
app = FastAPI(title="IMDB Sentiment Analysis API", version="1.0")

# Load model and vectorizer
print("Loading model and vectorizer...")
with open('models/logistic_regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('data/feature/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

print("Model and vectorizer loaded successfully!")

# Initialize preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


class ReviewInput(BaseModel):
    text: str


class PredictionOutput(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: dict


def preprocess_text(text):
    """
    Preprocess text for prediction
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize, remove stopwords, and lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)


@app.get("/")
def read_root():
    """
    Root endpoint - API information
    """
    return {
        "message": "IMDB Sentiment Analysis API",
        "version": "1.0",
        "endpoints": {
            "/predict": "POST - Predict sentiment of a review",
            "/health": "GET - Check API health status"
        }
    }


@app.get("/health")
def health_check():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }


@app.post("/predict", response_model=PredictionOutput)
def predict_sentiment(review: ReviewInput):
    """
    Predict sentiment of a movie review
    
    Args:
        review: ReviewInput object containing the text to analyze
        
    Returns:
        PredictionOutput with sentiment prediction and confidence
    """
    try:
        # Preprocess the text
        cleaned_text = preprocess_text(review.text)
        
        if not cleaned_text:
            raise HTTPException(status_code=400, detail="Text is empty after preprocessing")
        
        # Vectorize the text
        text_vectorized = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(text_vectorized)[0]
        probabilities = model.predict_proba(text_vectorized)[0]
        
        # Get sentiment label and confidence
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = float(probabilities[prediction])
        
        return PredictionOutput(
            text=review.text,
            sentiment=sentiment,
            confidence=confidence,
            probabilities={
                "negative": float(probabilities[0]),
                "positive": float(probabilities[1])
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
