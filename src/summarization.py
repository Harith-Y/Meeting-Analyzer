"""
Enhanced summary generation module with structured outputs for class lectures
"""
import requests
import json
from typing import Dict, Optional, Any, List
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import SUMMARY_MODELS, SUMMARY_PROMPTS, OPENROUTER_API_KEY
from src.logger import setup_logger

logger = setup_logger(__name__)


class SummaryGenerator:
    """Generate AI-powered summaries optimized for exam preparation"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.models = SUMMARY_MODELS
        self.prompts = SUMMARY_PROMPTS
    
    def generate_summary(
        self,
        transcript: str,
        summary_type: str = "class_lecture",
        model: str = "meta-llama/llama-3.2-3b-instruct:free",
        custom_instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate summary from transcript with specified format.
        
        Args:
            transcript: The full transcript text
            summary_type: Type of summary (class_lecture, brief_summary, detailed_notes)
            model: Model to use for summarization
            custom_instructions: Optional custom instructions to add to prompt
        
        Returns:
            dict: Summary results with text, metadata, and status
        """
        logger.info(f"Generating {summary_type} summary using {model}")
        
        # Validate API key
        if not self.api_key:
            error_msg = "OpenRouter API key not found. Please set OPENROUTER_API_KEY in .env file."
            logger.error(error_msg)
            return self._create_error_response(error_msg)
        
        # Validate model
        if model not in self.models:
            logger.warning(f"Model {model} not in config, using anyway...")
        
        # Get prompt template
        prompt_template = self.prompts.get(summary_type, self.prompts["class_lecture"])
        
        # Add custom instructions if provided
        if custom_instructions:
            prompt_template += f"\n\nAdditional Instructions: {custom_instructions}"
        
        # Create prompt
        prompt = prompt_template.format(transcript=transcript)
        
        try:
            # Call OpenRouter API with retry logic for rate limits
            logger.info("Calling OpenRouter API...")
            
            max_retries = 3
            retry_delay = 5  # seconds
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "https://github.com/yourusername/lecture-transcription",
                            "X-Title": "Class Lecture Transcription System"
                        },
                        json={
                            "model": model,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are an expert educational assistant helping students prepare for exams. Provide clear, comprehensive summaries that aid in studying and retention."
                                },
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            "temperature": 0.7,
                            "max_tokens": self.models.get(model, {}).get('max_tokens', 4096)
                        },
                        timeout=120
                    )
                    
                    # If we get a 429 (rate limit), retry
                    if response.status_code == 429 and attempt < max_retries - 1:
                        logger.warning(f"Rate limited (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay} seconds...")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    
                    # Exit retry loop if successful or final attempt
                    break
                    
                except requests.Timeout:
                    if attempt < max_retries - 1:
                        logger.warning(f"Timeout (attempt {attempt + 1}/{max_retries}), retrying...")
                        continue
                    raise
            
            # Check response
            if response.status_code == 200:
                response_data = response.json()
                summary_text = response_data["choices"][0]["message"]["content"]
                
                # Clean summary
                summary_text = self._clean_summary(summary_text)
                
                # Extract structured information
                structured_data = self._extract_structured_info(summary_text, summary_type)
                
                result = {
                    'success': True,
                    'summary': summary_text,
                    'summary_type': summary_type,
                    'structured_data': structured_data,
                    'model': model,
                    'model_name': self.models.get(model, {}).get('name', model),
                    'word_count': len(summary_text.split()),
                    'char_count': len(summary_text),
                    'timestamp': datetime.now().isoformat(),
                    'usage': response_data.get('usage', {})
                }
                
                logger.info(f"Summary generated successfully. Word count: {result['word_count']}")
                return result
            
            else:
                error_msg = f"API error {response.status_code}: {response.text}"
                logger.error(error_msg)
                return self._create_error_response(error_msg)
        
        except requests.Timeout:
            error_msg = "Request timed out after 120 seconds"
            logger.error(error_msg)
            return self._create_error_response(error_msg)
        
        except Exception as e:
            error_msg = f"Error generating summary: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return self._create_error_response(error_msg)
    
    def generate_key_points(self, transcript: str, max_points: int = 10) -> Dict[str, Any]:
        """
        Extract key points from transcript.
        
        Args:
            transcript: The transcript text
            max_points: Maximum number of key points to extract
        meta-llama/llama-3.1-8b-instruct
        Returns:
            dict: Key points and metadata
        """
        logger.info(f"Extracting up to {max_points} key points...")
        
        prompt = f"""Extract the {max_points} most important key points from this class lecture transcript.
Format as a numbered list with clear, concise points that students should remember.

Transcript:
{transcript}

Key Points:"""
        
        try:
            # Retry logic for rate limits
            max_retries = 3
            retry_delay = 5
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "google/gemini-2.0-flash-exp:free",
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.5,
                            "max_tokens": 1500
                        },
                        timeout=90
                    )
                    
                    if response.status_code == 429 and attempt < max_retries - 1:
                        logger.warning(f"Rate limited extracting key points (attempt {attempt + 1}/{max_retries}), retrying...")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    
                    if response.status_code == 200:
                        key_points_text = response.json()["choices"][0]["message"]["content"]
                        key_points_text = self._clean_summary(key_points_text)
                        
                        # Parse into list
                        key_points_list = self._parse_numbered_list(key_points_text)
                        
                        logger.info(f"Successfully extracted {len(key_points_list)} key points")
                        return {
                            'success': True,
                            'key_points': key_points_list,
                            'key_points_text': key_points_text,
                            'count': len(key_points_list),
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        error_msg = f"API error {response.status_code}: {response.text}"
                        logger.error(f"Key points extraction failed: {error_msg}")
                        return self._create_error_response(error_msg)
                
                except requests.Timeout:
                    if attempt < max_retries - 1:
                        logger.warning(f"Timeout extracting key points (attempt {attempt + 1}/{max_retries}), retrying...")
                        continue
                    raise
        
        except Exception as e:
            error_msg = f"Error extracting key points: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return self._create_error_response(error_msg)
    
    def generate_exam_questions(self, transcript: str, num_questions: int = 5) -> Dict[str, Any]:
        """
        Generate potential exam questions from transcript.
        
        Args:
            transcript: The transcript text
            num_questions: Number of questions to generate
        
        Returns:
            dict: Generated questions and metadata
        """
        logger.info(f"Generating {num_questions} potential exam questions...")
        
        prompt = f"""Based ONLY on the content from this class lecture transcript, generate {num_questions} exam questions with detailed answers.

CRITICAL REQUIREMENTS:
- Questions must be based ONLY on topics actually covered in this specific lecture
- Answers must reference specific concepts, examples, or explanations from the lecture
- Provide comprehensive, detailed answers (3-5 sentences minimum per answer)
- For multiple choice: include 4 options (A-D), then explain why the correct answer is right and why others are wrong

STRICT FORMAT - Follow this EXACTLY for each question:

**Question 1: Multiple Choice**

[Write a clear question based on a key concept from the lecture]

A) [First option]
B) [Second option]
C) [Third option]
D) [Fourth option]

**Answer:** [Letter]) [Restate correct answer]. [3-5 sentence explanation drawing from lecture content, explaining why this is correct and referencing what was discussed. Make it educational and comprehensive.]

---

**Question 2: Short Answer**

[Ask about an important concept, process, or relationship explained in the lecture]

**Answer:** [Write 4-6 sentences explaining the answer using information from the lecture. Include key terms, examples, or explanations that were mentioned. Make it detailed enough that a student studying from this will understand the topic.]

---

**Question 3: Essay**

[Ask a broader question requiring synthesis of multiple concepts from the lecture]

**Answer:** [Write 6-10 sentences providing a comprehensive answer. Connect multiple ideas from the lecture, explain relationships, provide context, and demonstrate deep understanding of the material covered.]

---

QUESTION MIX:
- 40% Multiple Choice questions (with detailed explanations)
- 30% Short Answer questions (4-6 sentence answers)
- 30% Essay questions (6-10 sentence comprehensive answers)

Lecture Transcript:
{transcript}

Now generate exactly {num_questions} questions following the format above:"""
        
        try:
            # Retry logic for rate limits
            max_retries = 3
            retry_delay = 5
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "meta-llama/llama-3.2-3b-instruct:free",
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.7,
                            "max_tokens": 4000
                        },
                        timeout=90
                    )
                    
                    if response.status_code == 429 and attempt < max_retries - 1:
                        logger.warning(f"Rate limited generating exam questions (attempt {attempt + 1}/{max_retries}), retrying...")
                        import time
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    
                    if response.status_code == 200:
                        questions_text = response.json()["choices"][0]["message"]["content"]
                        questions_text = self._clean_summary(questions_text)
                        
                        logger.info(f"Successfully generated {num_questions} exam questions")
                        return {
                            'success': True,
                            'questions': questions_text,
                            'count': num_questions,
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        error_msg = f"API error {response.status_code}: {response.text}"
                        logger.error(f"Exam questions generation failed: {error_msg}")
                        return self._create_error_response(error_msg)
                
                except requests.Timeout:
                    if attempt < max_retries - 1:
                        logger.warning(f"Timeout generating exam questions (attempt {attempt + 1}/{max_retries}), retrying...")
                        continue
                    raise
        
        except Exception as e:
            error_msg = f"Error generating exam questions: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return self._create_error_response(error_msg)
    
    def _clean_summary(self, text: str) -> str:
        """
        Clean up summary text by removing artifacts.
        
        Args:
            text: Raw summary text
        
        Returns:
            str: Cleaned text
        """
        # Remove common AI artifacts
        artifacts = [
            '<｜begin▁of▁sentence｜>',
            '<|begin_of_sentence|>',
            '<｜end▁of▁sentence｜>',
            '<|end_of_sentence|>',
            '<|im_start|>',
            '<|im_end|>',
        ]
        
        for artifact in artifacts:
            text = text.replace(artifact, '')
        
        # Clean up extra whitespace
        text = '\n'.join(line.strip() for line in text.split('\n'))
        text = text.strip()
        
        return text
    
    def _extract_structured_info(self, summary: str, summary_type: str) -> Dict[str, Any]:
        """
        Extract structured information from summary text.
        
        Args:
            summary: Summary text
            summary_type: Type of summary
        
        Returns:
            dict: Structured data extracted from summary
        """
        structured = {
            'has_topics': False,
            'has_key_concepts': False,
            'has_examples': False,
            'has_questions': False,
            'sections': []
        }
        
        # Simple detection of common sections
        summary_lower = summary.lower()
        
        if any(keyword in summary_lower for keyword in ['main topics', 'topics covered', '**main topics']):
            structured['has_topics'] = True
        
        if any(keyword in summary_lower for keyword in ['key concepts', 'definitions', 'important terms']):
            structured['has_key_concepts'] = True
        
        if any(keyword in summary_lower for keyword in ['examples', 'case studies', 'illustrations']):
            structured['has_examples'] = True
        
        if any(keyword in summary_lower for keyword in ['exam questions', 'potential questions', 'test questions']):
            structured['has_questions'] = True
        
        # Try to identify section headers (lines starting with ## or **)
        lines = summary.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('##') or (line.startswith('**') and line.endswith('**')):
                section_title = line.replace('##', '').replace('**', '').strip()
                if section_title:
                    structured['sections'].append(section_title)
        
        return structured
    
    def _parse_numbered_list(self, text: str) -> List[str]:
        """
        Parse numbered list from text.
        
        Args:
            text: Text containing numbered list
        
        Returns:
            list: List of points
        """
        points = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            # Match patterns like "1.", "1)", "1 -", etc.
            if line and any(line.startswith(f"{i}.") or line.startswith(f"{i})") for i in range(1, 100)):
                # Remove the number prefix
                for i in range(1, 100):
                    for prefix in [f"{i}.", f"{i})", f"{i} -", f"{i}-"]:
                        if line.startswith(prefix):
                            line = line[len(prefix):].strip()
                            break
                if line:
                    points.append(line)
        
        return points
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            'success': False,
            'summary': None,
            'errors': error_message,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_available_models(self) -> Dict[str, Dict]:
        """Get list of available summary models."""
        return self.models
    
    def get_summary_types(self) -> Dict[str, str]:
        """Get available summary types."""
        return {
            'class_lecture': 'Comprehensive study guide for exam preparation',
            'brief_summary': 'Quick overview of main topics and key takeaways',
            'detailed_notes': 'Detailed study notes with all concepts and examples'
        }


if __name__ == "__main__":
    # Test the summary generator
    generator = SummaryGenerator()
    print("Summary Generator initialized successfully")
    print("\nAvailable models:")
    for model_id, config in generator.get_available_models().items():
        print(f"  - {config['name']} ({model_id})")
    print("\nSummary types:")
    for type_id, description in generator.get_summary_types().items():
        print(f"  - {type_id}: {description}")
