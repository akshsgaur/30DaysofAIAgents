# agents/day02_text_summarizer/evaluation.py
"""
Evaluation Framework for LangChain Text Summarizer Agent
Day 2 of 30 Days of AI Agents Challenge

This module provides systematic evaluation of agent performance including:
- Ground truth dataset creation
- Multi-dimensional quality metrics
- Agent reasoning analysis
- Performance benchmarking
"""

import pandas as pd
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

from .agent import (
    TextSummarizerAgent, 
    SummaryRequest, 
    SummaryStrategy, 
    SummaryLength,
    SummaryResult
)


class SummarizerEvaluator:
    """
    Comprehensive evaluation framework for the Text Summarizer Agent.
    
    Features:
    - Ground truth dataset with expected outcomes
    - Multi-dimensional quality metrics
    - Agent reasoning quality analysis
    - Performance benchmarking and reporting
    """
    
    def __init__(self, agent: TextSummarizerAgent):
        """
        Initialize evaluator with agent instance.
        
        Args:
            agent: TextSummarizerAgent instance to evaluate
        """
        self.agent = agent
        self.evaluation_data = []
    
    def create_evaluation_dataset(self) -> pd.DataFrame:
        """
        Create a comprehensive ground truth dataset for evaluation.
        
        Returns:
            DataFrame with test cases and expected outcomes
        """
        test_cases = [
            {
                "test_id": "tech_001",
                "input_text": """
                Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving. The ideal characteristic of artificial intelligence is its ability to rationalize and take actions that have the best chance of achieving a specific goal. Machine learning, a subset of AI, provides systems the ability to automatically learn and improve from experience without being explicitly programmed.
                """.strip(),
                "strategy": SummaryStrategy.ABSTRACTIVE,
                "length": SummaryLength.SHORT,
                "expected_keywords": ["AI", "artificial intelligence", "machine learning", "human intelligence", "learning"],
                "expected_length_range": (50, 100),
                "content_type": "technical",
                "expected_tool": "abstractive_summarize"
            },
            {
                "test_id": "news_002", 
                "input_text": """
                Climate change refers to long-term shifts in global temperatures and weather patterns. While climate change is natural, human activities have been the main driver since the 1800s, primarily through the burning of fossil fuels like coal, oil and gas. These activities release greenhouse gases that trap heat in Earth's atmosphere, leading to global warming. The effects include rising sea levels, extreme weather events, ecosystem disruption, and threats to food security. Scientists emphasize the urgent need for action to reduce greenhouse gas emissions and transition to renewable energy sources.
                """.strip(),
                "strategy": SummaryStrategy.BULLET_POINTS,
                "length": SummaryLength.MEDIUM,
                "expected_keywords": ["climate change", "global warming", "greenhouse gases", "renewable energy"],
                "expected_length_range": (150, 250),
                "content_type": "news",
                "expected_tool": "bullet_points_summarize"
            },
            {
                "test_id": "research_003",
                "input_text": """
                A recent study published in Nature Medicine examined the effectiveness of telemedicine in managing chronic diseases. The research involved 2,500 patients across 15 healthcare facilities over 18 months. Key findings include: 1) 23% reduction in hospital readmissions for patients using telemedicine, 2) 89% patient satisfaction rate with virtual consultations, 3) 34% decrease in healthcare costs per patient, 4) Improved medication adherence rates from 67% to 85%. The study concludes that telemedicine represents a viable alternative to traditional in-person care for chronic disease management, with potential for widespread adoption in healthcare systems.
                """.strip(),
                "strategy": SummaryStrategy.KEY_TAKEAWAYS,
                "length": SummaryLength.LONG,
                "expected_keywords": ["telemedicine", "chronic diseases", "healthcare", "study", "findings"],
                "expected_length_range": (300, 500),
                "content_type": "research",
                "expected_tool": "key_takeaways_summarize"
            },
            {
                "test_id": "short_004",
                "input_text": """
                Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or qubits that can exist in multiple states simultaneously.
                """.strip(),
                "strategy": SummaryStrategy.EXTRACTIVE,
                "length": SummaryLength.SHORT,
                "expected_keywords": ["quantum computing", "qubits", "superposition"],
                "expected_length_range": (30, 80),
                "content_type": "short_technical",
                "expected_tool": "abstractive_summarize"  # Agent may choose different strategy
            },
            {
                "test_id": "business_005",
                "input_text": """
                The global e-commerce market has experienced unprecedented growth, reaching $4.9 trillion in 2021. Key drivers include increased internet penetration, mobile device adoption, and changing consumer behaviors accelerated by the COVID-19 pandemic. Major trends shaping the industry include social commerce, voice shopping, augmented reality experiences, and sustainability initiatives. Companies are investing heavily in logistics infrastructure, artificial intelligence for personalization, and omnichannel strategies to meet evolving customer expectations. Industry experts predict continued growth with the market expected to reach $7.4 trillion by 2025.
                """.strip(),
                "strategy": SummaryStrategy.ABSTRACTIVE,
                "length": SummaryLength.MEDIUM,
                "expected_keywords": ["e-commerce", "growth", "trends", "COVID-19", "AI"],
                "expected_length_range": (150, 250),
                "content_type": "business",
                "expected_tool": "abstractive_summarize"
            }
        ]
        
        return pd.DataFrame(test_cases)
    
    def evaluate_summary_quality(self, original_text: str, summary: str, expected_keywords: List[str]) -> Dict[str, float]:
        """
        Evaluate summary quality across multiple dimensions.
        
        Args:
            original_text: Original input text
            summary: Generated summary
            expected_keywords: Keywords that should appear in summary
            
        Returns:
            Dictionary of quality metrics
        """
        summary_lower = summary.lower()
        original_lower = original_text.lower()
        
        # Keyword coverage - what percentage of expected keywords appear
        keywords_found = sum(1 for keyword in expected_keywords if keyword.lower() in summary_lower)
        keyword_coverage = keywords_found / len(expected_keywords) if expected_keywords else 0
        
        # Compression ratio - summary length vs original length
        summary_words = len(summary.split())
        original_words = len(original_text.split())
        compression_ratio = summary_words / original_words if original_words > 0 else 0
        
        # Content preservation - important terms from original in summary
        original_words_set = set(word.lower() for word in original_text.split() if len(word) > 4)
        summary_words_set = set(word.lower() for word in summary.split())
        content_overlap = len(original_words_set.intersection(summary_words_set)) / len(original_words_set) if original_words_set else 0
        
        # Readability score (simple - based on sentence length and structure)
        sentences = re.split(r'[.!?]+', summary)
        sentences = [s.strip() for s in sentences if s.strip()]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        readability_score = max(0, min(1, (20 - avg_sentence_length) / 20)) if avg_sentence_length > 0 else 0
        
        # Combined relevance score
        relevance_score = (keyword_coverage * 0.4 + 
                          content_overlap * 0.3 + 
                          (1 - compression_ratio) * 0.2 + 
                          readability_score * 0.1)
        
        return {
            "keyword_coverage": keyword_coverage,
            "compression_ratio": compression_ratio,
            "content_overlap": content_overlap,
            "readability_score": readability_score,
            "relevance_score": min(relevance_score, 1.0),
            "summary_length": summary_words,
            "sentence_count": len(sentences)
        }
    
    def evaluate_agent_reasoning(self, reasoning_steps: List[str], expected_tool: str) -> Dict[str, float]:
        """
        Evaluate the quality of agent reasoning.
        
        Args:
            reasoning_steps: List of agent reasoning steps
            expected_tool: Expected tool the agent should use
            
        Returns:
            Dictionary of reasoning quality metrics
        """
        if not reasoning_steps:
            return {
                "reasoning_steps_count": 0,
                "tool_selection_accuracy": 0,
                "reasoning_quality": 0
            }
        
        # Count reasoning steps
        steps_count = len(reasoning_steps)
        
        # Check if agent used expected tool
        tool_used = None
        for step in reasoning_steps:
            if "Action:" in step:
                # Extract tool name from step
                tool_match = re.search(r'Action: ([^-]+)', step)
                if tool_match:
                    tool_used = tool_match.group(1).strip()
                    break
        
        tool_selection_accuracy = 1.0 if tool_used == expected_tool else 0.0
        
        # Evaluate reasoning quality based on patterns
        reasoning_quality = 0.0
        if any("analyze_document" in step for step in reasoning_steps):
            reasoning_quality += 0.3  # Started with analysis
        if any("Action:" in step for step in reasoning_steps):
            reasoning_quality += 0.4  # Used tools
        if any("Final Answer:" in step for step in reasoning_steps):
            reasoning_quality += 0.3  # Provided conclusion
        
        return {
            "reasoning_steps_count": steps_count,
            "tool_selection_accuracy": tool_selection_accuracy,
            "reasoning_quality": reasoning_quality,
            "tool_used": tool_used
        }
    
    def run_evaluation(self) -> pd.DataFrame:
        """
        Run comprehensive evaluation on the test dataset.
        
        Returns:
            DataFrame with detailed evaluation results
        """
        dataset = self.create_evaluation_dataset()
        results = []
        
        print(f"Running evaluation on {len(dataset)} test cases...")
        
        for idx, row in dataset.iterrows():
            print(f"Evaluating test case {idx + 1}/{len(dataset)}: {row['test_id']}")
            
            try:
                # Create summary request
                request = SummaryRequest(
                    text=row['input_text'],
                    strategy=row['strategy'],
                    length=row['length']
                )
                
                # Get summary from agent
                summary_result = self.agent.summarize_text(request)
                
                # Evaluate quality
                quality_metrics = self.evaluate_summary_quality(
                    row['input_text'],
                    summary_result.summary,
                    row['expected_keywords']
                )
                
                # Evaluate reasoning
                reasoning_metrics = self.evaluate_agent_reasoning(
                    summary_result.agent_reasoning or [],
                    row['expected_tool']
                )
                
                # Check length compliance
                expected_min, expected_max = row['expected_length_range']
                length_compliant = expected_min <= summary_result.word_count <= expected_max
                length_difference = abs(summary_result.word_count - (expected_min + expected_max) / 2)
                
                # Compile results
                result = {
                    "test_id": row['test_id'],
                    "content_type": row['content_type'],
                    "strategy_requested": row['strategy'].value,
                    "strategy_used": summary_result.strategy_used.value,
                    "length_requested": row['length'].value,
                    "expected_length": f"{expected_min}-{expected_max}",
                    "actual_length": summary_result.word_count,
                    "length_compliant": length_compliant,
                    "length_difference": length_difference,
                    "processing_time": summary_result.processing_time,
                    "chunks_processed": summary_result.chunks_processed,
                    
                    # Quality metrics
                    "keyword_coverage": quality_metrics["keyword_coverage"],
                    "compression_ratio": quality_metrics["compression_ratio"],
                    "content_overlap": quality_metrics["content_overlap"],
                    "readability_score": quality_metrics["readability_score"],
                    "relevance_score": quality_metrics["relevance_score"],
                    
                    # Reasoning metrics
                    "reasoning_steps": reasoning_metrics["reasoning_steps_count"],
                    "tool_selection_accuracy": reasoning_metrics["tool_selection_accuracy"],
                    "reasoning_quality": reasoning_metrics["reasoning_quality"],
                    "tool_used": reasoning_metrics["tool_used"],
                    
                    # Summary preview
                    "summary_preview": summary_result.summary[:100] + "..." if len(summary_result.summary) > 100 else summary_result.summary,
                    "full_summary": summary_result.summary
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error evaluating {row['test_id']}: {str(e)}")
                # Add error result
                error_result = {
                    "test_id": row['test_id'],
                    "content_type": row['content_type'],
                    "strategy_requested": row['strategy'].value,
                    "error": str(e),
                    "length_compliant": False,
                    "relevance_score": 0.0,
                    "tool_selection_accuracy": 0.0,
                    "reasoning_quality": 0.0
                }
                results.append(error_result)
        
        return pd.DataFrame(results)
    
    def generate_evaluation_report(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            results_df: Results from run_evaluation()
            
        Returns:
            Dictionary with evaluation summary and insights
        """
        total_tests = len(results_df)
        successful_tests = len(results_df[~results_df.get('error', pd.Series(dtype=bool)).notna()])
        
        if successful_tests == 0:
            return {"error": "No successful tests to analyze"}
        
        success_rate = successful_tests / total_tests
        
        # Calculate averages for successful tests
        successful_df = results_df[~results_df.get('error', pd.Series(dtype=bool)).notna()]
        
        avg_metrics = {
            "relevance_score": successful_df['relevance_score'].mean(),
            "keyword_coverage": successful_df['keyword_coverage'].mean(),
            "length_compliance_rate": successful_df['length_compliant'].mean(),
            "tool_selection_accuracy": successful_df['tool_selection_accuracy'].mean(),
            "reasoning_quality": successful_df['reasoning_quality'].mean(),
            "avg_processing_time": successful_df['processing_time'].mean(),
            "avg_reasoning_steps": successful_df['reasoning_steps'].mean()
        }
        
        # Performance by content type
        content_performance = {}
        for content_type in successful_df['content_type'].unique():
            subset = successful_df[successful_df['content_type'] == content_type]
            content_performance[content_type] = {
                "count": len(subset),
                "avg_relevance": subset['relevance_score'].mean(),
                "avg_tool_accuracy": subset['tool_selection_accuracy'].mean()
            }
        
        # Identify areas for improvement
        improvements = []
        if avg_metrics["relevance_score"] < 0.7:
            improvements.append("Relevance score below 0.7 - consider improving content focus")
        if avg_metrics["length_compliance_rate"] < 0.8:
            improvements.append("Length compliance below 80% - review length control mechanisms")
        if avg_metrics["tool_selection_accuracy"] < 0.7:
            improvements.append("Tool selection accuracy below 70% - agent reasoning may need refinement")
        if avg_metrics["keyword_coverage"] < 0.6:
            improvements.append("Keyword coverage below 60% - enhance keyword preservation")
        
        return {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": success_rate,
                "evaluation_timestamp": datetime.now().isoformat()
            },
            "performance_metrics": avg_metrics,
            "content_type_performance": content_performance,
            "improvement_recommendations": improvements,
            "top_performing_strategies": self._analyze_strategy_performance(successful_df),
            "detailed_results": results_df.to_dict('records')
        }
    
    def _analyze_strategy_performance(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze which strategies perform best."""
        strategy_performance = {}
        for strategy in df['strategy_requested'].unique():
            subset = df[df['strategy_requested'] == strategy]
            strategy_performance[strategy] = subset['relevance_score'].mean()
        return strategy_performance