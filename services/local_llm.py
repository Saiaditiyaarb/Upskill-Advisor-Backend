"""
Local LLM utilities for Upskill Advisor.

Provides a consistent interface for using local language models instead of cloud-based APIs.
Supports various local models through Hugging Face Transformers and LangChain integration.

Design choices:
- Uses Hugging Face Transformers for local model inference
- Provides both direct and LangChain-compatible interfaces
- Implements caching and batching for performance
- Graceful fallback mechanisms when models are unavailable
- Configurable model parameters through environment settings
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union
from functools import lru_cache

from core.config import get_settings

logger = logging.getLogger("local_llm")


class LocalLLM:
    """Local language model wrapper with LangChain compatibility."""

    def __init__(self, model_name: Optional[str] = None, cache_dir: Optional[str] = None):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.local_llm_model
        self.cache_dir = cache_dir or self.settings.hf_home
        self.max_length = self.settings.max_sequence_length

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir
        os.environ["HF_HOME"] = self.cache_dir

        self._pipeline = None
        self._tokenizer = None
        self._model = None

    def _load_model(self):
        """Lazy load the model and tokenizer."""
        if self._pipeline is not None:
            return

        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
            import torch

            logger.info(f"Loading local LLM: {self.model_name}")

            # Determine device
            device = 0 if torch.cuda.is_available() else -1

            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                padding_side="left"
            )

            # Add pad token if it doesn't exist
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Load model with appropriate settings
            model_kwargs = {
                "cache_dir": self.cache_dir,
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "low_cpu_mem_usage": True,
            }

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )

            # Create pipeline
            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                device=device,
                max_length=self.max_length,
                do_sample=True,
                temperature=0.1,
                top_p=0.9,
                pad_token_id=self._tokenizer.eos_token_id,
                return_full_text=False
            )

            logger.info(f"Successfully loaded local LLM: {self.model_name}")

        except Exception as e:
            logger.error(f"Failed to load local LLM {self.model_name}: {e}")
            self._pipeline = None
            raise

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.1) -> str:
        """Generate text using the local model."""
        if not self.settings.use_local_llm:
            raise RuntimeError("Local LLM is disabled in configuration")

        self._load_model()

        if self._pipeline is None:
            raise RuntimeError("Local LLM model failed to load")

        try:
            # Generate response
            result = self._pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id
            )

            if result and len(result) > 0:
                generated_text = result[0]["generated_text"].strip()
                return generated_text
            else:
                return ""

        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return ""

    def predict(self, text: str) -> str:
        """LangChain-compatible prediction method."""
        return self.generate(text)

    async def apredict(self, text: str) -> str:
        """Async prediction method for LangChain compatibility."""
        return self.generate(text)

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """LangChain-compatible call method."""
        if isinstance(inputs, dict) and "text" in inputs:
            result = self.generate(inputs["text"])
            return {"text": result}
        elif isinstance(inputs, str):
            result = self.generate(inputs)
            return {"text": result}
        else:
            raise ValueError("Invalid input format for LocalLLM")


@lru_cache(maxsize=1)
def get_local_llm() -> Optional[LocalLLM]:
    """Get cached local LLM instance."""
    settings = get_settings()

    if not settings.use_local_llm:
        logger.info("Local LLM disabled in configuration")
        return None

    try:
        llm = LocalLLM()
        # Test load to ensure it works
        llm._load_model()
        return llm
    except Exception as e:
        logger.warning(f"Failed to initialize local LLM: {e}")
        return None


class LocalLLMChain:
    """Simple chain implementation for local LLM with prompt templates."""

    def __init__(self, llm: LocalLLM, prompt_template: str):
        self.llm = llm
        self.prompt_template = prompt_template

    def run(self, **kwargs) -> str:
        """Run the chain with the given inputs."""
        try:
            prompt = self.prompt_template.format(**kwargs)
            return self.llm.generate(prompt)
        except Exception as e:
            logger.error(f"LocalLLMChain execution failed: {e}")
            return ""

    async def arun(self, **kwargs) -> str:
        """Async version of run."""
        return self.run(**kwargs)

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """LangChain-compatible invoke method."""
        result = self.run(**inputs)
        return {"text": result}

    async def ainvoke(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Async version of invoke."""
        result = await self.arun(**inputs)
        return {"text": result}


def create_local_llm_chain(prompt_template: str) -> Optional[LocalLLMChain]:
    """Create a local LLM chain with the given prompt template."""
    llm = get_local_llm()
    if llm is None:
        return None
    return LocalLLMChain(llm, prompt_template)


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from generated text, handling common formatting issues."""
    import json
    import re

    # Try to find JSON in the text
    json_patterns = [
        r'\{.*\}',  # Simple JSON object
        r'\[.*\]',  # JSON array
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                # Clean up common issues
                cleaned = match.strip()
                # Remove markdown code blocks
                cleaned = re.sub(r'```json\s*', '', cleaned)
                cleaned = re.sub(r'```\s*$', '', cleaned)

                return json.loads(cleaned)
            except json.JSONDecodeError:
                continue

    # Try parsing the entire text as JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    logger.warning("Could not extract valid JSON from generated text")
    return None


class LocalJsonOutputParser:
    """JSON output parser for local LLM responses."""

    def parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from text output."""
        return extract_json_from_text(text)

    def __call__(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """LangChain-compatible call method."""
        text = inputs.get("text", "")
        result = self.parse(text)
        return result or {}


# Fallback implementations for when local models are not available
class FallbackLLM:
    """Fallback LLM that returns simple template-based responses."""

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a simple fallback response."""
        if "learning plan" in prompt.lower():
            return '''{"plan": [{"course_id": "fallback-1", "why": "Fallback recommendation - configure local LLM for better results", "order": 1, "estimated_weeks": 4}], "timeline": {"total_weeks": 4}, "gap_map": {}, "notes": "This is a fallback response. Please configure a local LLM model for better recommendations."}'''
        elif "job description" in prompt.lower():
            return '''{"job_title": "Software Developer", "required_skills": ["Programming", "Problem Solving"], "years_of_experience_min": 2, "company": "Unknown", "location": "Unknown", "job_type": "full-time"}'''
        else:
            return "Fallback response - local LLM not available"

    def predict(self, text: str) -> str:
        return self.generate(text)

    async def apredict(self, text: str) -> str:
        return self.generate(text)


def get_llm_with_fallback() -> Union[LocalLLM, FallbackLLM]:
    """Get local LLM with fallback to simple template responses."""
    llm = get_local_llm()
    if llm is not None:
        return llm

    logger.warning("Using fallback LLM - install transformers and configure local model for better results")
    return FallbackLLM()