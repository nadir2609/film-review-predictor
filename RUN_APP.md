# Running the IMDB Sentiment Analysis Web Application

## Quick Start

### 1. Start the FastAPI Backend (Terminal 1)
```bash
# Activate virtual environment
.\env\Scripts\Activate.ps1

# Run the API
python api.py
```
The API will be available at: http://localhost:8000

### 2. Start the Streamlit Frontend (Terminal 2)
```bash
# Activate virtual environment
.\env\Scripts\Activate.ps1

# Run the Streamlit app
streamlit run app.py
```
The web interface will open automatically in your browser at: http://localhost:8501

## API Endpoints

- **GET /** - API information
- **GET /health** - Health check
- **POST /predict** - Predict sentiment
  ```json
  {
    "text": "Your movie review here"
  }
  ```

## Features

- 🎬 Real-time sentiment prediction
- 📊 Confidence scores and probability distribution
- 💡 Example reviews for testing
- 🎨 Beautiful, user-friendly interface
- ⚡ Fast API backend with automatic documentation at http://localhost:8000/docs

## Model Performance

- Accuracy: 88.98%
- Precision: 87.71%
- Recall: 90.65%
- ROC-AUC: 95.64%
