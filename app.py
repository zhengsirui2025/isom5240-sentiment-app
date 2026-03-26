"""
ISOM5240 Group Project - Sentiment Analysis App
Customer Feedback Classification with Deep Learning

This application uses two pipelines:
1. Pipeline 1: Sentiment Analysis (Fine-tuned Model)
2. Pipeline 2: Text-to-Speech (Pre-trained Model)
"""

import streamlit as st
from transformers import pipeline
import torch
import time

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1F4E79;
        text-align: center;
        margin-bottom: 1rem;
    }
    .result-positive {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .result-negative {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;·
    }
    .result-neutral {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_sentiment_model():
    """Load the fine-tuned sentiment analysis model"""
    model_path = "zzzsr/sentiment-analysis-finetuned"
    try:
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_path,
            tokenizer=model_path,
            device=0 if torch.cuda.is_available() else -1
        )
        return sentiment_pipeline
    except Exception as e:
        st.error(f"Error loading sentiment model: {e}")
        return None

@st.cache_resource
def load_tts_model():
    """Load the text-to-speech model"""
    try:
        tts_pipeline = pipeline(
            "text-to-speech",
            model="suno/bark-small",
            device=0 if torch.cuda.is_available() else -1
        )
        return tts_pipeline
    except Exception as e:
        st.error(f"Error loading TTS model: {e}")
        return None

def main():
    # Header
    st.markdown("<h1 class='main-header'>Customer Feedback Sentiment Analysis</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.info("""
        This application analyzes customer feedback sentiment 
        using a fine-tuned deep learning model.
        
        **Pipelines:**
        - Pipeline 1: Sentiment Classification
        - Pipeline 2: Text-to-Speech Output
        """)
        
        st.header("Model Information")
        st.markdown("""
        **Sentiment Model:**
        - yj2773/hinglish11k-sentiment-analysis
        - Fine-tuned with custom dataset
        - Accuracy: 90.1%
        
        **TTS Model:**
        - suno/bark-small
        - Pre-trained
        """)
    
    # Load models
    with st.spinner("Loading models..."):
        sentiment_model = load_sentiment_model()
        tts_model = load_tts_model()
    
    if sentiment_model is None:
        st.error("Failed to load sentiment model. Please check the model path.")
        return
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Customer Feedback")
        user_input = st.text_area(
            "Input text here:",
            height=150,
            placeholder="Type or paste customer feedback here..."
        )
        
        analyze_button = st.button("Analyze Sentiment", type="primary")
    
    with col2:
        st.subheader("Settings")
        enable_tts = st.checkbox("Enable Audio Output", value=True)
        show_confidence = st.checkbox("Show Confidence Score", value=True)
    
    # Process input
    if analyze_button and user_input.strip():
        with st.spinner("Analyzing..."):
            start_time = time.time()
            
            # Get sentiment prediction
            result = sentiment_model(user_input)[0]
            
            processing_time = time.time() - start_time
            
            label = result['label'].lower()
            confidence = result['score']
            
            # Display results
            st.markdown("---")
            st.subheader("Analysis Results")
            
            # Create result display
            if label == 'positive':
                st.markdown(f"""
                <div class='result-positive'>
                    <h3>Positive Sentiment ✓</h3>
                    <p>The customer feedback indicates satisfaction.</p>
                </div>
                """, unsafe_allow_html=True)
            elif label == 'negative':
                st.markdown(f"""
                <div class='result-negative'>
                    <h3>Negative Sentiment ✗</h3>
                    <p>The customer feedback indicates dissatisfaction.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-neutral'>
                    <h3>Neutral Sentiment ○</h3>
                    <p>The customer feedback is neutral or mixed.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sentiment", label.capitalize())
            with col2:
                if show_confidence:
                    st.metric("Confidence", f"{confidence:.2%}")
            with col3:
                st.metric("Processing Time", f"{processing_time:.2f}s")
            
            # TTS output
            if enable_tts and tts_model:
                st.markdown("---")
                st.subheader("Audio Output")
                
                tts_text = f"The sentiment of your feedback is {label}."
                
                with st.spinner("Generating audio..."):
                    try:
                        audio_output = tts_model(tts_text)
                        st.audio(audio_output['audio'], sample_rate=audio_output['sampling_rate'])
                    except Exception as e:
                        st.warning(f"Could not generate audio: {e}")
    
    elif analyze_button:
        st.warning("Please enter some text to analyze.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>ISOM5240 Group Project | Sentiment Analysis App</p>
        <p>Developed by Group 01 | March 2026</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
