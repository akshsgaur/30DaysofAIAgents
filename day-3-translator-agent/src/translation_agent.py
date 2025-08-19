"""Main Language Translator Agent"""

from typing import List, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage

from .models import TranslationResult, TranslationQuality, SUPPORTED_LANGUAGES
from .language_detector import LanguageDetector
from .translation_memory import TranslationMemory

class TranslatorAgent:
    def __init__(self, openai_api_key: str, model: str = "gpt-4"):
        self.llm = ChatOpenAI(api_key=openai_api_key, model=model, temperature=0.1)
        self.language_detector = LanguageDetector(self.llm)
        self.translation_memory = TranslationMemory()
        self.memory = ConversationBufferWindowMemory(k=10, return_messages=True)
        self.tools = self._create_tools()
        self.agent = self._create_agent()
    
    def _create_tools(self) -> List[Tool]:
        def translate_text_tool(query: str) -> str:
            try:
                parts = query.split(" | ")
                if len(parts) < 2:
                    return "Error: Use format 'text | target_language'"
                
                text = parts[0].strip()
                target_lang = parts[1].strip().lower()
                source_lang = parts[2].strip().lower() if len(parts) > 2 else None
                
                result = self.translate_single(text, target_lang, source_lang)
                return self._format_result(result)
            except Exception as e:
                return f"Translation error: {str(e)}"
        
        def detect_language_tool(text: str) -> str:
            try:
                lang_code, confidence = self.language_detector.detect_language(text)
                lang_name = SUPPORTED_LANGUAGES.get(lang_code, "Unknown")
                return f"Language: {lang_name} ({lang_code}) - {confidence:.1%} confidence"
            except Exception as e:
                return f"Detection error: {str(e)}"
        
        def list_languages_tool(query: str = "") -> str:
            return "Supported languages:\n" + "\n".join(f"{name} ({code})" for code, name in SUPPORTED_LANGUAGES.items())
        
        return [
            Tool(name="translate_text", description="Translate text. Format: 'text | target_language'", func=translate_text_tool),
            Tool(name="detect_language", description="Detect language of text", func=detect_language_tool),
            Tool(name="list_languages", description="List supported languages", func=list_languages_tool)
        ]
    
    def _create_agent(self) -> AgentExecutor:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Language Translator Agent. 
            Provide accurate translations, detect languages, and handle cultural context.
            Available tools: translate_text, detect_language, list_languages"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, memory=self.memory, verbose=True, max_iterations=3)
    
    def translate_single(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> TranslationResult:
        """Translate a single text"""
        start_time = datetime.now()
        
        # Auto-detect source language if not provided
        if not source_lang:
            source_lang, _ = self.language_detector.detect_language(text)
        
        # Check cache
        cached = self.translation_memory.get_cached_translation(text, source_lang, target_lang)
        if cached:
            return cached
        
        # Translate
        target_lang_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)
        translation_prompt = f"""
        Translate this text to {target_lang_name}:
        "{text}"
        
        Provide only the translation, maintaining tone and context.
        """
        
        response = self.llm.invoke([HumanMessage(content=translation_prompt)])
        translated_text = response.content.strip()
        
        # Create result
        result = TranslationResult(
            original_text=text,
            translated_text=translated_text,
            source_language=source_lang,
            target_language=target_lang,
            confidence_score=0.85,
            quality_assessment=TranslationQuality.GOOD,
            timestamp=datetime.now(),
            processing_time=(datetime.now() - start_time).total_seconds()
        )
        
        # Cache result
        self.translation_memory.store_translation(result)
        return result
    
    def translate_batch(self, texts: List[str], target_lang: str, source_lang: Optional[str] = None) -> List[TranslationResult]:
        """Translate multiple texts"""
        return [self.translate_single(text, target_lang, source_lang) for text in texts]
    
    def _format_result(self, result: TranslationResult) -> str:
        return f"""
Translation: {result.translated_text}
Quality: {result.quality_assessment.value}
Time: {result.processing_time:.2f}s
        """
    
    def process_request(self, user_input: str) -> str:
        """Process user request through agent"""
        try:
            response = self.agent.invoke({"input": user_input})
            return response["output"]
        except Exception as e:
            return f"Error: {str(e)}"