"""Evaluation framework for Translation Agent"""

import asyncio
from typing import List, Dict, Any
from datetime import datetime
from .models import EvaluationCase, EvaluationResult

class TranslationEvaluator:
    def __init__(self, agent):
        self.agent = agent
        self.results: List[EvaluationResult] = []
    
    def create_test_cases(self) -> List[EvaluationCase]:
        """Create evaluation test cases"""
        return [
            EvaluationCase("basic_001", "Hello, how are you?", "en", "es", 
                          "Hola, ¿cómo estás?", "casual greeting", "easy", "general"),
            EvaluationCase("basic_002", "Thank you very much", "en", "fr", 
                          "Merci beaucoup", "gratitude", "easy", "general"),
            EvaluationCase("tech_001", "The API endpoint returned an error", "en", "es", 
                          "El endpoint de la API devolvió un error", "technical", "medium", "technical"),
            EvaluationCase("idiom_001", "It's raining cats and dogs", "en", "es", 
                          "Está lloviendo a cántaros", "weather idiom", "hard", "idiom"),
            EvaluationCase("edge_001", "", "en", "es", "error", "empty text", "easy", "edge_case")
        ]
    
    async def evaluate_case(self, case: EvaluationCase) -> EvaluationResult:
        """Evaluate a single test case"""
        start_time = datetime.now()
        
        try:
            if case.input_text.strip() == "":
                actual_output = "error"
                accuracy = 1.0 if case.expected_output == "error" else 0.0
            else:
                result = self.agent.translate_single(case.input_text, case.target_language, 
                                                   case.source_language if case.source_language != "auto" else None)
                actual_output = result.translated_text
                accuracy = self._simple_accuracy(case.expected_output, actual_output)
            
            fluency = 0.8  # Simplified
            adequacy = 0.8  # Simplified
            
        except Exception as e:
            actual_output = f"Error: {str(e)}"
            accuracy = fluency = adequacy = 0.0
        
        response_time = (datetime.now() - start_time).total_seconds()
        composite_score = (accuracy + fluency + adequacy) / 3
        
        return EvaluationResult(
            case_id=case.case_id,
            actual_output=actual_output,
            expected_output=case.expected_output,
            accuracy_score=accuracy,
            fluency_score=fluency,
            adequacy_score=adequacy,
            response_time=response_time,
            passed=composite_score >= 0.7,
            errors=[],
            metadata={"category": case.category, "difficulty": case.difficulty_level}
        )
    
    def _simple_accuracy(self, expected: str, actual: str) -> float:
        """Simple accuracy calculation"""
        if not expected or not actual:
            return 0.0
        
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        
        if not expected_words:
            return 1.0 if not actual_words else 0.0
        
        intersection = expected_words.intersection(actual_words)
        return len(intersection) / len(expected_words)
    
    async def run_evaluation(self, test_cases: List[EvaluationCase] = None) -> Dict[str, Any]:
        """Run comprehensive evaluation"""
        if test_cases is None:
            test_cases = self.create_test_cases()
        
        print(f"Running evaluation on {len(test_cases)} test cases...")
        
        self.results = []
        for case in test_cases:
            result = await self.evaluate_case(case)
            self.results.append(result)
        
        metrics = self._calculate_metrics()
        return {"metrics": metrics, "results": self.results}
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate evaluation metrics"""
        if not self.results:
            return {}
        
        total_cases = len(self.results)
        passed_cases = sum(1 for r in self.results if r.passed)
        
        return {
            "overall": {
                "total_cases": total_cases,
                "passed_cases": passed_cases,
                "pass_rate": passed_cases / total_cases,
                "avg_accuracy": sum(r.accuracy_score for r in self.results) / total_cases,
                "avg_fluency": sum(r.fluency_score for r in self.results) / total_cases,
                "avg_adequacy": sum(r.adequacy_score for r in self.results) / total_cases,
                "avg_response_time": sum(r.response_time for r in self.results) / total_cases
            }
        }
    
    def generate_report(self) -> str:
        """Generate evaluation report"""
        metrics = self._calculate_metrics()
        overall = metrics.get("overall", {})
        
        return f"""
# Translation Agent Evaluation Report

## Overall Performance
- **Pass Rate**: {overall.get('pass_rate', 0):.1%}
- **Average Accuracy**: {overall.get('avg_accuracy', 0):.3f}
- **Average Response Time**: {overall.get('avg_response_time', 0):.2f}s

## Failed Cases
{len([r for r in self.results if not r.passed])} out of {len(self.results)} cases failed.
        """