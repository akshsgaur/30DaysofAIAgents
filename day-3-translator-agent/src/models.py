"""Core data models for Translation Agent"""

from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from typing import List, Dict, Any

class TranslationQuality(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

@dataclass
class TranslationResult:
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence_score: float
    quality_assessment: TranslationQuality
    timestamp: datetime
    processing_time: float

@dataclass
class EvaluationCase:
    case_id: str
    input_text: str
    source_language: str
    target_language: str
    expected_output: str
    context: str
    difficulty_level: str
    category: str

@dataclass
class EvaluationResult:
    case_id: str
    actual_output: str
    expected_output: str
    accuracy_score: float
    fluency_score: float
    adequacy_score: float
    response_time: float
    passed: bool
    errors: List[str]
    metadata: Dict[str, Any]

# Language mappings
SUPPORTED_LANGUAGES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "hi": "Hindi",
    "th": "Thai", "vi": "Vietnamese", "tr": "Turkish", "pl": "Polish",
    "nl": "Dutch", "sv": "Swedish", "da": "Danish", "no": "Norwegian"
}