# 🌐 Language Translator Agent

An intelligent AI-powered translation system built with LangChain and Streamlit, featuring advanced evaluation capabilities, cultural context awareness, and comprehensive translation memory.

## 🎯 Features

### 🚀 **Core Capabilities**
- **Multi-language Translation**: Support for 20+ languages including major European, Asian, and Middle Eastern languages
- **Intelligent Language Detection**: Automatic source language identification with confidence scoring
- **Cultural Context Awareness**: Proper handling of idioms, cultural references, and formal/informal registers
- **Batch Translation**: Process multiple texts simultaneously with consistency guarantees

### 🧠 **Advanced AI Features**
- **ReAct Agent Pattern**: Reasoning + Acting for intelligent decision-making
- **Translation Memory**: Automatic caching for consistency and performance
- **Quality Assessment**: Multi-dimensional evaluation (accuracy, fluency, adequacy)
- **Tool Orchestration**: Seamless integration of multiple specialized tools

### 📊 **Evaluation Framework**
- **Systematic Testing**: 30+ comprehensive test cases covering various scenarios
- **Performance Metrics**: Pass rate, accuracy, fluency, and response time tracking
- **Data-Driven Improvements**: Automated identification of improvement areas
- **Custom Test Cases**: Easy addition of domain-specific evaluation scenarios

### 💻 **User Experience**
- **Interactive Chat Interface**: Natural conversation flow with memory
- **Batch Processing UI**: File upload and download capabilities
- **Real-time Analytics**: Performance monitoring and cache statistics
- **Professional Streamlit Interface**: Modern, responsive design

## 🛠️ Tech Stack

- **AI Framework**: LangChain with OpenAI GPT-4/3.5-turbo
- **Frontend**: Streamlit with interactive components
- **Backend**: Python with async support
- **Memory Systems**: Conversation buffer + Translation cache
- **Evaluation**: Custom async framework with multiple metrics

## 📦 Installation

### Prerequisites
- Python 3.8+
- OpenAI API key
- Virtual environment (recommended)

### Quick Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd language-translator-agent

# Create virtual environment
python -m venv translator
source translator/bin/activate  # On Windows: translator\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install langchain langchain-openai langchain-community streamlit openai pandas python-dotenv pydantic

# Set up environment variables
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env

# Run the application
streamlit run translator_app.py