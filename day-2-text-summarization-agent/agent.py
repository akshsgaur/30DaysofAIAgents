# agents/day02_text_summarizer/agent.py
"""
Text Summarizer Agent - Working Version with ReAct Pattern
Day 2 of 30 Days of AI Agents Challenge

This implementation uses a simple ReAct agent that works reliably
without function calling compatibility issues.
"""

import os
import tempfile
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json

# Use stable LangChain components
from langchain.llms import OpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.callbacks import BaseCallbackHandler
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Document Processing
import docx2txt
import PyPDF2


class SummaryStrategy(Enum):
    """Enumeration of available summarization strategies."""
    EXTRACTIVE = "extractive"
    ABSTRACTIVE = "abstractive" 
    BULLET_POINTS = "bullet_points"
    KEY_TAKEAWAYS = "key_takeaways"


class SummaryLength(Enum):
    """Enumeration of summary length options."""
    SHORT = "short"      # 50-100 words
    MEDIUM = "medium"    # 150-250 words
    LONG = "long"        # 300-500 words


@dataclass
class SummaryRequest:
    """Data class for summarization requests."""
    text: str
    strategy: SummaryStrategy
    length: SummaryLength
    custom_instructions: str = ""


@dataclass
class SummaryResult:
    """Data class for summarization results."""
    summary: str
    word_count: int
    strategy_used: SummaryStrategy
    processing_time: float
    chunks_processed: int
    original_word_count: int = 0
    compression_ratio: float = 0.0
    agent_reasoning: List[str] = None


class DocumentProcessor:
    """Utility class for document processing."""
    
    @staticmethod
    def extract_text_from_file(uploaded_file) -> str:
        """Extract text from uploaded file."""
        file_type = uploaded_file.type
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        
        try:
            if file_type == "application/pdf":
                return DocumentProcessor._extract_pdf_text(tmp_file_path)
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                return DocumentProcessor._extract_docx_text(tmp_file_path)
            elif file_type == "text/plain":
                return DocumentProcessor._extract_txt_text(tmp_file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
        finally:
            os.unlink(tmp_file_path)
    
    @staticmethod
    def _extract_pdf_text(file_path: str) -> str:
        """Extract text from PDF file."""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    
    @staticmethod
    def _extract_docx_text(file_path: str) -> str:
        """Extract text from DOCX file."""
        return docx2txt.process(file_path).strip()
    
    @staticmethod
    def _extract_txt_text(file_path: str) -> str:
        """Extract text from TXT file."""
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    return file.read().strip()
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode text file")


class SummarizerTools:
    """Collection of tools for the summarization agent."""
    
    def __init__(self, llm, text_splitter):
        self.llm = llm
        self.text_splitter = text_splitter
        self.current_text = ""
        self.processing_stats = {}
    
    def analyze_document_tool(self, text: str) -> str:
        """Analyze document to determine optimal summarization approach."""
        word_count = len(text.split())
        char_count = len(text)
        needs_chunking = char_count > 4000
        
        # Store for other tools
        self.current_text = text
        self.processing_stats = {
            "word_count": word_count,
            "char_count": char_count,
            "needs_chunking": needs_chunking
        }
        
        analysis = {
            "word_count": word_count,
            "character_count": char_count,
            "estimated_chunks": len(self.text_splitter.split_text(text)) if needs_chunking else 1,
            "processing_approach": "chunked" if needs_chunking else "direct"
        }
        
        return json.dumps(analysis, indent=2)
    
    def abstractive_summarize_tool(self, length: str, custom_instructions: str = "") -> str:
        """Create abstractive summary by rewriting content."""
        if not self.current_text:
            return "Error: No text has been analyzed yet."
        
        word_targets = {
            "short": "50-100 words",
            "medium": "150-250 words", 
            "long": "300-500 words"
        }
        
        prompt = f"""
        Create an abstractive summary by rewriting the content in your own words.
        
        Target length: {word_targets.get(length, word_targets['medium'])}
        {custom_instructions}
        
        Requirements:
        1. Rewrite in your own words while preserving meaning
        2. Focus on main ideas and key insights
        3. Maintain logical flow and coherence
        4. Include important facts and conclusions
        
        Text:
        {self.current_text}
        
        Abstractive Summary:
        """
        
        return self.llm(prompt)
    
    def bullet_points_summarize_tool(self, length: str, custom_instructions: str = "") -> str:
        """Create bullet point summary."""
        if not self.current_text:
            return "Error: No text has been analyzed yet."
        
        bullet_counts = {
            "short": "3-5 bullet points",
            "medium": "6-8 bullet points",
            "long": "9-12 bullet points"
        }
        
        prompt = f"""
        Create a bullet point summary of the main ideas and key information.
        
        Format: {bullet_counts.get(length, bullet_counts['medium'])}
        {custom_instructions}
        
        Requirements:
        1. Use clear, concise bullet points
        2. Start each point with a bullet (•)
        3. Focus on distinct main ideas
        4. Include important facts and insights
        
        Text:
        {self.current_text}
        
        Bullet Point Summary:
        """
        
        return self.llm(prompt)
    
    def key_takeaways_summarize_tool(self, length: str, custom_instructions: str = "") -> str:
        """Create numbered key takeaways summary."""
        if not self.current_text:
            return "Error: No text has been analyzed yet."
        
        takeaway_counts = {
            "short": "3-5 key takeaways",
            "medium": "6-8 key takeaways",
            "long": "9-12 key takeaways"
        }
        
        prompt = f"""
        Extract the key takeaways and insights from the text.
        
        Format: {takeaway_counts.get(length, takeaway_counts['medium'])}
        {custom_instructions}
        
        Requirements:
        1. Number each takeaway (1., 2., 3., etc.)
        2. Focus on actionable insights and conclusions
        3. Include important facts and findings
        4. Present in order of importance
        
        Text:
        {self.current_text}
        
        Key Takeaways:
        """
        
        return self.llm(prompt)


class AgentCallbackHandler(BaseCallbackHandler):
    """Custom callback handler to track agent reasoning."""
    
    def __init__(self):
        self.reasoning_steps = []
    
    def on_agent_action(self, action, **kwargs):
        """Track agent actions/tool calls."""
        self.reasoning_steps.append(f"Action: {action.tool} - {action.tool_input}")
    
    def on_agent_finish(self, finish, **kwargs):
        """Track agent completion."""
        output = finish.return_values.get('output', '')
        preview = output[:100] + "..." if len(output) > 100 else output
        self.reasoning_steps.append(f"Final Answer: {preview}")


class TextSummarizerAgent:
    """
    ReAct Text Summarizer Agent using stable OpenAI completion API.
    
    This version avoids function calling issues by using the ReAct pattern
    with standard text completion rather than chat completion with functions.
    """
    
    def __init__(self, openai_api_key: str, model: str = "gpt-3.5-turbo-instruct", verbose: bool = False):
        """Initialize the agent with stable OpenAI completion API."""
        self.openai_api_key = openai_api_key
        self.verbose = verbose
        
        # Use stable OpenAI completion LLM instead of ChatOpenAI
        self.llm = OpenAI(
            openai_api_key=openai_api_key,
            model_name=model,
            temperature=0.3,
            max_tokens=1000
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200
        )
        
        # Initialize tools
        self.tools_handler = SummarizerTools(self.llm, self.text_splitter)
        self.tools = self._create_tools()
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=False  # Use string format for compatibility
        )
        
        # Create agent
        self.agent_executor = self._create_agent()
    
    def _create_tools(self) -> List[Tool]:
        """Create the agent's tools."""
        return [
            Tool(
                name="analyze_document",
                description="Analyze a document to understand its structure. Always use this first.",
                func=self.tools_handler.analyze_document_tool
            ),
            Tool(
                name="abstractive_summarize",
                description="Create an abstractive summary by rewriting content in new words.",
                func=lambda inputs: self._parse_summarize_input(inputs, self.tools_handler.abstractive_summarize_tool)
            ),
            Tool(
                name="bullet_points_summarize",
                description="Create a bullet point summary highlighting main ideas.",
                func=lambda inputs: self._parse_summarize_input(inputs, self.tools_handler.bullet_points_summarize_tool)
            ),
            Tool(
                name="key_takeaways_summarize",
                description="Extract numbered key takeaways and insights.",
                func=lambda inputs: self._parse_summarize_input(inputs, self.tools_handler.key_takeaways_summarize_tool)
            )
        ]
    
    def _parse_summarize_input(self, inputs: str, tool_func) -> str:
        """Parse summarization tool inputs."""
        try:
            if inputs.startswith('{'):
                parsed = json.loads(inputs)
                length = parsed.get('length', 'medium')
                custom_instructions = parsed.get('custom_instructions', '')
            else:
                parts = inputs.split('|')
                length = parts[0].strip() if parts else 'medium'
                custom_instructions = parts[1].strip() if len(parts) > 1 else ''
            
            return tool_func(length, custom_instructions)
        except:
            return tool_func('medium', inputs)
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent executor with ReAct pattern."""
        
        # Simple ReAct prompt template that works with completion models
        agent_prompt = PromptTemplate.from_template("""
You are an expert text summarization agent. Your job is to analyze documents and create high-quality summaries.

Available Tools:
{tools}

Tool Names: {tool_names}

Process:
1. ALWAYS start by analyzing the document with analyze_document tool
2. Based on the analysis, choose the most appropriate summarization strategy
3. Use the selected tool with proper length and any custom instructions

Use this format:
Question: the input question
Thought: think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Question: {input}
{agent_scratchpad}
""")
        
        # Create ReAct agent (works with completion models)
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=agent_prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=self.verbose,
            max_iterations=5,
            handle_parsing_errors=True
        )
    
    def summarize_text(self, request: SummaryRequest) -> SummaryResult:
        """Main method to summarize text using the agent."""
        start_time = datetime.now()
        callback_handler = AgentCallbackHandler()
        
        agent_input = f"""
        Please summarize the following text using the {request.strategy.value} strategy with {request.length.value} length.
        
        Custom instructions: {request.custom_instructions}
        
        Text to summarize:
        {request.text}
        """
        
        try:
            result = self.agent_executor.invoke(
                {"input": agent_input},
                config={"callbacks": [callback_handler]}
            )
            summary = result["output"]
        except Exception as e:
            summary = f"Error during agent summarization: {str(e)}"
        
        processing_time = (datetime.now() - start_time).total_seconds()
        original_word_count = len(request.text.split())
        summary_word_count = len(summary.split())
        compression_ratio = summary_word_count / original_word_count if original_word_count > 0 else 0
        
        return SummaryResult(
            summary=summary,
            word_count=summary_word_count,
            strategy_used=request.strategy,
            processing_time=processing_time,
            chunks_processed=len(self.text_splitter.split_text(request.text)),
            original_word_count=original_word_count,
            compression_ratio=compression_ratio,
            agent_reasoning=callback_handler.reasoning_steps
        )
    
    def load_document_from_file(self, uploaded_file) -> str:
        """Load document using DocumentProcessor utility."""
        return DocumentProcessor.extract_text_from_file(uploaded_file)
    
    def batch_summarize(self, texts: List[str], strategy: SummaryStrategy, length: SummaryLength) -> List[SummaryResult]:
        """Batch summarization using the agent."""
        results = []
        for text in texts:
            request = SummaryRequest(text=text, strategy=strategy, length=length)
            result = self.summarize_text(request)
            results.append(result)
        return results
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "agent_type": "LangChain ReAct Text Summarizer Agent",
            "day": 2,
            "framework": "LangChain Agent Framework",
            "reasoning_pattern": "ReAct (Reasoning + Acting)",
            "tools": [tool.name for tool in self.tools],
            "capabilities": {
                "strategies": [s.value for s in SummaryStrategy],
                "lengths": [l.value for l in SummaryLength],
                "file_types": ["PDF", "DOCX", "TXT"],
                "agent_reasoning": True,
                "memory": True,
                "tool_selection": "Automatic"
            },
            "model": getattr(self.llm, 'model_name', 'gpt-3.5-turbo-instruct')
        }