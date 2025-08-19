"""Translation memory for caching and consistency"""

from typing import Dict, Optional
from .models import TranslationResult

class TranslationMemory:
    def __init__(self):
        self.memory: Dict[str, TranslationResult] = {}
    
    def get_translation_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Generate unique key for translation pair"""
        return f"{source_lang}:{target_lang}:{hash(text.lower().strip())}"
    
    def get_cached_translation(self, text: str, source_lang: str, target_lang: str) -> Optional[TranslationResult]:
        """Retrieve cached translation if available"""
        key = self.get_translation_key(text, source_lang, target_lang)
        return self.memory.get(key)
    
    def store_translation(self, result: TranslationResult):
        """Store translation result in memory"""
        key = self.get_translation_key(result.original_text, result.source_language, result.target_language)
        self.memory[key] = result
    
    def clear_cache(self):
        """Clear all cached translations"""
        self.memory.clear()
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        return {
            "total_translations": len(self.memory),
            "language_pairs": len(set(f"{r.source_language}->{r.target_language}" for r in self.memory.values()))
        }