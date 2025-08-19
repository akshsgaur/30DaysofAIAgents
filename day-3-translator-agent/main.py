"""
Main application entry point for Language Translator Agent
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now import and run the app
from src.ui.app import create_streamlit_app

if __name__ == "__main__":
    create_streamlit_app()