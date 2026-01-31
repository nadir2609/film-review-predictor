# 🎬 IMDB Sentiment Analysis

A complete end-to-end machine learning project for sentiment analysis of IMDB movie reviews, featuring a full data pipeline with DVC, a trained Logistic Regression model, and a web application built with FastAPI and Streamlit.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.53-red.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-green.svg)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/sklearn-latest-orange.svg)](https://scikit-learn.org/)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline](#pipeline)
- [Web Application](#web-application)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)

## 🎯 Overview

This project implements a sentiment analysis system for IMDB movie reviews using Natural Language Processing (NLP) and Machine Learning. The system classifies movie reviews as either **positive** or **negative** with high accuracy.

### Key Highlights:
- 📊 **50,000** IMDB reviews dataset
- 🎯 **88.98%** accuracy
- 🚀 **Production-ready** API and web interface
- 📦 **DVC pipeline** for reproducible ML workflows
- 🧪 Comprehensive text preprocessing and feature engineering

## ✨ Features

- **Complete ML Pipeline**: Data ingestion → Processing → Feature Engineering → Model Training → Evaluation
- **DVC Integration**: Reproducible data and model versioning
- **REST API**: FastAPI backend for predictions
- **Web Interface**: Beautiful Streamlit UI for easy interaction
- **Real-time Predictions**: Instant sentiment analysis with confidence scores
- **Preprocessing Pipeline**: HTML removal, stopword filtering, lemmatization
- **TF-IDF Vectorization**: Advanced feature extraction with 151,000+ features

## 📊 Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 88.98% |
| Precision | 87.71% |
| Recall    | 90.65% |
| F1-Score  | 89.16% |
| ROC-AUC   | 95.64% |

**Confusion Matrix:**
```
TN: 3492  |  FP: 508
FN: 374   |  TP: 3626
```

## 📁 Project Structure

```
IMBD project/
├── data/
│   ├── raw/              # Raw train/test data (DVC tracked)
│   ├── processed/        # Cleaned and preprocessed data
│   └── feature/          # TF-IDF features and vectorizer
├── models/               # Trained models (DVC tracked)
│   └── logistic_regression_model.pkl
├── results/              # Evaluation metrics (JSON)
├── src/
│   ├── data_ingestion.py       # Load and split data
│   ├── data_processing.py      # Text preprocessing
│   ├── feature_engineering.py  # TF-IDF vectorization
│   ├── model_building.py       # Model training
│   └── model_evaluation.py     # Model evaluation
├── Notebook/
│   └── experiment.ipynb        # Jupyter notebook experiments
├── api.py                # FastAPI backend
├── app.py                # Streamlit frontend
├── dvc.yaml              # DVC pipeline definition
├── requirements.txt      # Python dependencies
├── start_api.bat         # Windows script to start API
├── start_streamlit.bat   # Windows script to start web app
└── README.md
```

## 🚀 Installation

### Prerequisites
- Python 3.12+
- Git
- DVC (optional, for pipeline reproduction)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
```

2. **Create virtual environment**
```bash
python -m venv env
```

3. **Activate virtual environment**
- Windows:
  ```bash
  .\env\Scripts\activate
  ```
- Linux/Mac:
  ```bash
  source env/bin/activate
  ```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Download NLTK data**
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## 💻 Usage

### Option 1: Using Scripts (Windows)

**Start the API:**
```bash
.\start_api.bat
```
API runs at: http://localhost:8000

**Start the Web App:**
```bash
.\start_streamlit.bat
```
Web app opens at: http://localhost:8501

### Option 2: Manual Start

**Terminal 1 - Start API:**
```bash
.\env\Scripts\activate
python api.py
```

**Terminal 2 - Start Streamlit:**
```bash
.\env\Scripts\activate
streamlit run app.py
```

### Option 3: Run Pipeline Scripts Individually

```bash
# Activate environment
.\env\Scripts\activate

# Run individual pipeline steps
python src/data_ingestion.py
python src/data_processing.py
python src/feature_engineering.py
python src/model_building.py
python src/model_evaluation.py
```

## 🔄 Pipeline

The project uses **DVC** for pipeline management. To reproduce the entire pipeline:

1. **Install DVC** (if not already installed):
```bash
pip install dvc
```

2. **Run the pipeline**:
```bash
dvc repro
```

This will execute all stages:
- `data_ingestion`: Load and split IMDB dataset
- `data_processing`: Clean and preprocess text
- `feature_engineering`: Extract TF-IDF features
- `model_building`: Train Logistic Regression model
- `model_evaluation`: Evaluate and save metrics

## 🌐 Web Application

### API Endpoints

**GET /** - API information
```json
{
  "message": "IMDB Sentiment Analysis API",
  "version": "1.0"
}
```

**POST /predict** - Predict sentiment
```json
// Request
{
  "text": "This movie was absolutely fantastic!"
}

// Response
{
  "text": "This movie was absolutely fantastic!",
  "sentiment": "Positive",
  "confidence": 0.95,
  "probabilities": {
    "negative": 0.05,
    "positive": 0.95
  }
}
```

**GET /health** - Health check
```json
{
  "status": "healthy",
  "model_loaded": true,
  "vectorizer_loaded": true
}
```

### API Documentation

FastAPI provides automatic interactive documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🛠️ Technologies Used

### Core
- **Python 3.12**: Programming language
- **scikit-learn**: Machine learning library
- **pandas**: Data manipulation
- **numpy**: Numerical computing

### NLP
- **NLTK**: Natural language processing
- **TF-IDF Vectorizer**: Feature extraction

### Pipeline & Versioning
- **DVC**: Data and model versioning
- **Git**: Code versioning

### Web & API
- **FastAPI**: REST API framework
- **Streamlit**: Web application framework
- **uvicorn**: ASGI server
- **Pydantic**: Data validation

### Development
- **Jupyter**: Interactive development
- **pytest**: Testing (optional)

## 🔍 How It Works

1. **Data Ingestion**: Load 50K IMDB reviews and split into train/test sets
2. **Preprocessing**:
   - Convert to lowercase
   - Remove HTML tags and URLs
   - Remove special characters
   - Remove stopwords
   - Lemmatize words
3. **Feature Engineering**: Convert text to TF-IDF vectors
4. **Model Training**: Train Logistic Regression classifier
5. **Evaluation**: Calculate metrics and save results
6. **Deployment**: Serve model via FastAPI, interface via Streamlit

## 📈 Future Improvements

- [ ] Add more ML models (LSTM, BERT, Transformers)
- [ ] Implement model comparison dashboard
- [ ] Add unit tests and CI/CD
- [ ] Deploy to cloud (AWS/GCP/Heroku)
- [ ] Add batch prediction endpoint
- [ ] Implement feedback loop for continuous learning
- [ ] Add multilingual support

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- IMDB dataset for providing the movie reviews
- scikit-learn community for excellent ML tools
- Streamlit and FastAPI teams for amazing frameworks

## 📧 Contact

For questions or feedback, please open an issue on GitHub or contact me directly.

---

⭐ **If you found this project helpful, please give it a star!** ⭐
