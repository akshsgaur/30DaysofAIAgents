"""Streamlit UI for Translation Agent"""

import streamlit as st
import pandas as pd
import asyncio
import sys
import os

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.translation_agent import TranslatorAgent
from src.evaluator import TranslationEvaluator
from src.models import SUPPORTED_LANGUAGES

def create_streamlit_app():
    st.set_page_config(page_title="🌐 Language Translator Agent", layout="wide")
    
    st.title("🌐 Language Translator Agent")
    st.markdown("*AI-powered translation with evaluation*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        model = st.selectbox("Model", ["gpt-4", "gpt-3.5-turbo"])
    
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key.")
        return
    
    # Initialize agent
    if 'agent' not in st.session_state:
        with st.spinner("Initializing agent..."):
            st.session_state.agent = TranslatorAgent(openai_api_key, model)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📝 Batch", "🔍 Detection", "🧪 Evaluation"])
    
    # Chat Tab
    with tab1:
        st.subheader("Chat Translation")
        
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask for translation..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                response = st.session_state.agent.process_request(prompt)
                st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    # Batch Tab
    with tab2:
        st.subheader("Batch Translation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            texts = st.text_area("Texts (one per line)", height=200)
            target_lang = st.selectbox("Target Language", 
                                     options=list(SUPPORTED_LANGUAGES.keys()),
                                     format_func=lambda x: f"{SUPPORTED_LANGUAGES[x]} ({x})")
        
        with col2:
            if st.button("🚀 Translate Batch"):
                if texts.strip():
                    text_list = [line.strip() for line in texts.split('\n') if line.strip()]
                    
                    with st.spinner("Translating..."):
                        results = st.session_state.agent.translate_batch(text_list, target_lang)
                    
                    # Display results
                    df_data = []
                    for result in results:
                        df_data.append({
                            "Original": result.original_text,
                            "Translation": result.translated_text,
                            "Quality": result.quality_assessment.value
                        })
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)
    
    # Detection Tab
    with tab3:
        st.subheader("Language Detection")
        
        detection_text = st.text_area("Enter text for detection", height=150)
        
        if st.button("🔍 Detect Language"):
            if detection_text.strip():
                with st.spinner("Detecting..."):
                    lang_code, confidence = st.session_state.agent.language_detector.detect_language(detection_text)
                    lang_name = SUPPORTED_LANGUAGES.get(lang_code, "Unknown")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Language", lang_name)
                with col2:
                    st.metric("Code", lang_code)
                with col3:
                    st.metric("Confidence", f"{confidence:.1%}")
    
    # Evaluation Tab
    with tab4:
        st.subheader("Agent Evaluation")
        
        if st.button("🧪 Run Evaluation"):
            with st.spinner("Running evaluation..."):
                evaluator = TranslationEvaluator(st.session_state.agent)
                
                # Run evaluation (simplified for demo)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                results = loop.run_until_complete(evaluator.run_evaluation())
                loop.close()
                
                # Display results
                metrics = results["metrics"]["overall"]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pass Rate", f"{metrics['pass_rate']:.1%}")
                with col2:
                    st.metric("Accuracy", f"{metrics['avg_accuracy']:.3f}")
                with col3:
                    st.metric("Response Time", f"{metrics['avg_response_time']:.2f}s")
                
                # Show report
                st.text_area("Evaluation Report", evaluator.generate_report(), height=300)

if __name__ == "__main__":
    create_streamlit_app()