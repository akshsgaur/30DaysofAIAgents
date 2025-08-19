"""Language detection functionality"""

import json
import re
from typing import Tuple
from langchain.schema import HumanMessage
from .models import SUPPORTED_LANGUAGES

class LanguageDetector:
    def __init__(self, llm):
        self.llm = llm
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language of input text with confidence score"""
        if not text.strip():
            return "unknown", 0.0
            
        prompt = f"""
        Detect the language of this text: "{text}"
        
        Respond in JSON format:
        {{"language_code": "ISO_code", "confidence": 0.95}}
        
        Use only these codes: {list(SUPPORTED_LANGUAGES.keys())}
        """
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get("language_code", "unknown"), result.get("confidence", 0.0)
        except Exception as e:
            print(f"Language detection error: {e}")
        
        return "unknown", 0.0