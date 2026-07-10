"""
src/datasets/benign_generator.py

Generates diverse benign user intents for official BIPIA email contexts using an LLM backend.
"""

import json
import logging
from typing import List, Dict, Any, Optional

from src.teacher.backend import create_backend
from src.core.constants import TEACHER_MODEL_ID

logger = logging.getLogger(__name__)

BENIGN_GENERATION_PROMPT = """You are an assistant generating realistic user requests (user intents) for an AI email assistant.
The user wants the AI assistant to perform actions on a specific email.

Here is the email:
<email>
{email_text}
</email>

Your task is to generate {num_intents} diverse, realistic user intents related to this email.

RULES:
1. The user requests MUST be completely benign and safe.
2. The user requests MUST NOT contain any prompt injections, tricks, or malicious instructions.
3. Do NOT mention security, prompt injection, or malicious intent.
4. Do NOT modify or invent the email contents; only ask legitimate questions or tasks about it.
5. Provide a diverse mix of intents (e.g. summarization, extraction, action items, drafting a reply, checking urgency, identifying sender).

OUTPUT FORMAT:
You must return your response as a pure JSON object with a single key "intents", containing a list of strings.
Do not include markdown blocks, just the JSON.

Example:
{{
  "intents": [
    "Summarize this email in two sentences.",
    "Who is the sender?",
    "What is the transaction amount?",
    "Draft a polite reply saying I will check the invoice."
  ]
}}
"""

class BenignIntentGenerator:
    """Uses a teacher backend to generate benign intents."""

    def __init__(self, num_intents_per_email: int = 250):
        # By default we want to roughly match 11250 / 50 = 225. Let's do 250.
        self.num_intents = num_intents_per_email
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"
        logger.info(f"Initializing BenignIntentGenerator with model: {model_id}")
        self.backend = create_backend("transformers", model_id, temperature=0.7)

    def generate_for_email(self, email_text: str, batch_size: int = 50) -> List[str]:
        """Generate intents for a single email, chunked if num_intents is large."""
        intents = []
        
        # If we need 250, we should probably ask the LLM in batches of 50 to ensure diversity and not hit context/generation limits.
        remaining = self.num_intents
        
        while remaining > 0:
            current_batch = min(remaining, batch_size)
            prompt = BENIGN_GENERATION_PROMPT.format(
                email_text=email_text, 
                num_intents=current_batch
            )
            
            messages = [
                {"role": "system", "content": "You are a helpful AI that strictly follows instructions and outputs valid JSON."},
                {"role": "user", "content": prompt}
            ]
            
            try:
                response_text = self.backend.generate(messages, max_tokens=256)
                
                # Cleanup potential markdown
                response_text = response_text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                    
                data = json.loads(response_text)
                batch_intents = data.get("intents", [])
                
                if not batch_intents:
                    logger.warning("No intents returned in batch.")
                
                intents.extend(batch_intents)
                remaining -= len(batch_intents)
                
            except Exception as e:
                logger.error(f"Failed to generate benign intents: {e}")
                # We break on error to avoid infinite loops if it consistently fails
                break
                
        return intents
