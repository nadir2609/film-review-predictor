"""
Streamlit Application for IMDB Sentiment Analysis
User-friendly interface for sentiment prediction
"""

import streamlit as st
import requests
import json

# API endpoint
API_URL = "http://localhost:8000/predict"

# Page configuration
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextArea textarea {
        font-size: 16px;
    }
    .positive {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .negative {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.markdown("""
Welcome! Enter a movie review below and our AI model will predict whether it's **positive** or **negative**.
""")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This application uses a **Logistic Regression** model trained on 50,000 IMDB movie reviews.
    
    **Model Performance:**
    - Accuracy: 88.98%
    - Precision: 87.71%
    - Recall: 90.65%
    - ROC-AUC: 95.64%
    
    **How it works:**
    1. Enter your review
    2. Click "Analyze Sentiment"
    3. Get instant prediction
    """)
    
    st.markdown("---")
    st.markdown("**API Status**")
    
    # Check API health
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        if response.status_code == 200:
            st.success("✅ API is running")
        else:
            st.error("❌ API error")
    except:
        st.error("❌ API is not running")
        st.info("Please start the API with: `python api.py`")

# Main content
st.markdown("### Enter Your Movie Review")

# Text input
review_text = st.text_area(
    "Type or paste a movie review here:",
    height=150,
    placeholder="Example: This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout..."
)

# Example reviews
with st.expander("📝 Try Example Reviews"):
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Positive Example"):
            review_text = "This movie was absolutely fantastic! The acting was superb, the cinematography was breathtaking, and the plot kept me engaged from start to finish. I highly recommend it to everyone!"
            st.rerun()
    
    with col2:
        if st.button("Negative Example"):
            review_text = "What a waste of time! The plot was predictable, the acting was terrible, and I found myself checking my watch every five minutes. I wouldn't recommend this to anyone."
            st.rerun()

# Predict button
if st.button("🔍 Analyze Sentiment", type="primary"):
    if not review_text.strip():
        st.warning("⚠️ Please enter a review first!")
    else:
        with st.spinner("Analyzing sentiment..."):
            try:
                # Make API request
                response = requests.post(
                    API_URL,
                    json={"text": review_text},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display result
                    sentiment = result['sentiment']
                    confidence = result['confidence'] * 100
                    
                    # Create result box
                    if sentiment == "Positive":
                        st.markdown(f"""
                        <div class="positive">
                            <h3>😊 Positive Review</h3>
                            <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="negative">
                            <h3>😞 Negative Review</h3>
                            <p><strong>Confidence:</strong> {confidence:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display probability distribution
                    st.markdown("### Probability Distribution")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Negative",
                            f"{result['probabilities']['negative'] * 100:.2f}%"
                        )
                    
                    with col2:
                        st.metric(
                            "Positive",
                            f"{result['probabilities']['positive'] * 100:.2f}%"
                        )
                    
                    # Progress bars
                    st.progress(result['probabilities']['positive'])
                    
                else:
                    st.error(f"❌ Error: {response.json().get('detail', 'Unknown error')}")
            
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Please make sure the API is running.")
                st.info("Start the API with: `python api.py`")
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out. Please try again.")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit + FastAPI | IMDB Sentiment Analysis Model</p>
</div>
""", unsafe_allow_html=True)
