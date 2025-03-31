"""
Client for interacting with Ollama API to generate text with LLMs
"""

import json
import re
import time
from typing import Any, Dict, List, Optional

import requests

from .logger import get_logger

# Create logger
logger = get_logger(__name__)


def filter_thinking(response_text: str, show_thinking: bool = False) -> str:
    """Filter out <think> blocks from the response unless explicitly shown

    Args:
        response_text: The raw response text from the model
        show_thinking: Whether to include thinking sections in the output

    Returns:
        Filtered response text
    """
    if show_thinking:
        return response_text

    # Use regex to remove sections between <think> and </think> tags
    # This handles multiline content and potential nested tags
    pattern = r"<think>.*?</think>\s*"
    filtered_text = re.sub(pattern, "", response_text, flags=re.DOTALL)

    # Trim any extra whitespace from the beginning and end
    return filtered_text.strip()


class OllamaAPIClient:
    """Client for interacting with Ollama API"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize the Ollama API client

        Args:
            base_url: Base URL for the Ollama API (default: http://localhost:11434)
        """
        self.base_url = base_url.rstrip("/")
        self.default_model = "qwq:latest"
        logger.info(f"Initialized Ollama API client with base URL: {self.base_url}")

    def _handle_request_error(self, error: Exception, endpoint: str) -> None:
        """Handle request errors

        Args:
            error: The exception that occurred
            endpoint: The API endpoint that was being accessed

        Raises:
            RuntimeError: Always raised with details about the error
        """
        error_msg = f"Error calling Ollama API endpoint {endpoint}: {str(error)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models

        Returns:
            List of available models
        """
        endpoint = "/api/tags"
        try:
            response = requests.get(f"{self.base_url}{endpoint}")
            response.raise_for_status()
            return response.json().get("models", [])
        except Exception as e:
            self._handle_request_error(e, endpoint)

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        stream: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate text using the Ollama API

        Args:
            prompt: The prompt to generate text from
            model: The model to use (default: None, uses default_model)
            system: System prompt to use (default: None)
            temperature: Temperature for generation (default: 0.7)
            top_p: Top-p sampling value (default: 0.9)
            max_tokens: Maximum number of tokens to generate (default: 1024)
            stream: Whether to stream the response (default: False)
            **kwargs: Additional parameters to pass to the API

        Returns:
            Dictionary containing the response
        """
        model = model or self.default_model

        if stream:
            return self._stream_generate(
                prompt=prompt,
                model=model,
                system=system,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                **kwargs,
            )

        return self._generate(
            prompt=prompt,
            model=model,
            system=system,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            **kwargs,
        )

    def _generate(
        self,
        prompt: str,
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate text without streaming

        Args:
            prompt: The prompt to generate text from
            model: The model to use
            system: System prompt to use (default: None)
            temperature: Temperature for generation (default: 0.7)
            top_p: Top-p sampling value (default: 0.9)
            max_tokens: Maximum number of tokens to generate (default: 1024)
            **kwargs: Additional parameters to pass to the API

        Returns:
            Dictionary containing the response
        """
        endpoint = "/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
            "stream": False,  # Explicitly set stream to False
        }

        # Add system prompt if provided
        if system:
            payload["system"] = system

        # Add any additional parameters
        for key, value in kwargs.items():
            if key != "stream":  # Don't override stream parameter
                payload[key] = value

        try:
            logger.debug(f"Sending request to Ollama API: {model}, prompt length: {len(prompt)}")
            start_time = time.time()
            response = requests.post(f"{self.base_url}{endpoint}", json=payload)
            response.raise_for_status()
            elapsed_time = time.time() - start_time
            logger.debug(f"Ollama response received in {elapsed_time:.2f}s")

            result = response.json()
            return result
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            logger.error(f"Response content: {response.text[:200]}...")
            self._handle_request_error(e, endpoint)
        except Exception as e:
            self._handle_request_error(e, endpoint)

    def _stream_generate(
        self,
        prompt: str,
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 1024,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate text with streaming

        Args:
            prompt: The prompt to generate text from
            model: The model to use
            system: System prompt to use (default: None)
            temperature: Temperature for generation (default: 0.7)
            top_p: Top-p sampling value (default: 0.9)
            max_tokens: Maximum number of tokens to generate (default: 1024)
            **kwargs: Additional parameters to pass to the API

        Returns:
            Dictionary containing the complete response
        """
        endpoint = "/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
            "stream": True,
        }

        # Add system prompt if provided
        if system:
            payload["system"] = system

        # Add any additional parameters
        for key, value in kwargs.items():
            payload[key] = value

        try:
            logger.debug(f"Sending streaming request to Ollama API: {model}")
            start_time = time.time()

            full_response = ""
            with requests.post(f"{self.base_url}{endpoint}", json=payload, stream=True) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        chunk_text = chunk.get("response", "")
                        full_response += chunk_text

                        # Check if done
                        if chunk.get("done", False):
                            break

            elapsed_time = time.time() - start_time
            logger.debug(f"Ollama streaming response completed in {elapsed_time:.2f}s")

            # Return a structured response similar to non-streaming
            return {
                "model": model,
                "response": full_response,
                "done": True,
            }
        except Exception as e:
            self._handle_request_error(e, endpoint)

    def generate_caption(
        self,
        subject: str,
        species: Optional[str] = None,
        gender: Optional[str] = None,
        background: Optional[str] = None,
        style: Optional[str] = None,
        temperature: float = 0.7,
        model: Optional[str] = None,
        show_thinking: bool = False,
    ) -> str:
        """Generate a caption based on parameters

        Args:
            subject: The main subject/character for the image
            species: The species of the character (default: None)
            gender: The gender of the character (default: None)
            background: The background/setting (default: None)
            style: The style of the caption (default: None)
            temperature: Temperature for generation (default: 0.7)
            model: The model to use (default: None, uses default_model)
            show_thinking: Whether to include model's thinking process in the output

        Returns:
            Generated caption
        """
        # Build prompt components
        prompt_parts = []
        prompt_parts.append(f"Subject: {subject}")

        if species:
            prompt_parts.append(f"Species: {species}")
        if gender:
            prompt_parts.append(f"Gender: {gender}")
        if background:
            prompt_parts.append(f"Background: {background}")
        if style:
            prompt_parts.append(f"Style: {style}")

        # Create the main prompt
        param_text = "\n".join(prompt_parts)

        system = (
            "You are a helpful creative assistant that generates detailed, vivid captions "
            "for text-to-image generation. Your captions should be descriptive, cohesive, "
            "and suitable for high-quality image generation. Focus on visual elements, "
            "composition, lighting, mood, and artistic style."
        )

        prompt = (
            f"Generate a detailed, descriptive caption for an image with the following parameters:\n\n"
            f"{param_text}\n\n"
            f"The caption should be detailed and vivid, focusing on visual elements. "
            f"Format your response as a single paragraph without explanations or extra text."
        )

        # Generate the caption
        logger.debug(f"Generating caption for: {subject}")
        response = self.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            model=model or self.default_model,
        )

        # Extract and return the caption
        caption = response.get("response", "").strip()
        caption = filter_thinking(caption, show_thinking)
        logger.debug(f"Generated caption of length: {len(caption)}")

        return caption

    def generate_nsfw_caption(
        self,
        subject: str,
        species: Optional[str] = None,
        gender: Optional[str] = None,
        background: Optional[str] = None,
        style: Optional[str] = None,
        nsfw_intensity: str = "moderate",
        temperature: float = 0.7,
        model: Optional[str] = None,
        show_thinking: bool = False,
    ) -> str:
        """Generate an NSFW caption based on parameters

        Args:
            subject: The main subject/character for the image
            species: The species of the character (default: None)
            gender: The gender of the character (default: None)
            background: The background/setting (default: None)
            style: The style of the caption (default: None)
            nsfw_intensity: Level of NSFW content (mild, moderate, explicit) (default: moderate)
            temperature: Temperature for generation (default: 0.7)
            model: The model to use (default: None, uses default_model)
            show_thinking: Whether to include model's thinking process in the output

        Returns:
            Generated NSFW caption
        """
        # Build prompt components
        prompt_parts = []
        prompt_parts.append(f"Subject: {subject}")

        if species:
            prompt_parts.append(f"Species: {species}")
        if gender:
            prompt_parts.append(f"Gender: {gender}")
        if background:
            prompt_parts.append(f"Background: {background}")
        if style:
            prompt_parts.append(f"Style: {style}")

        # Add NSFW intensity
        prompt_parts.append(f"NSFW Level: {nsfw_intensity}")

        # Create the main prompt
        param_text = "\n".join(prompt_parts)

        system = (
            "You are a creative assistant that generates detailed, vivid captions "
            "for adult/NSFW furry art generation. Your captions should be descriptive, "
            "cohesive, and appropriate for the specified NSFW level. Focus on visual elements, "
            "composition, poses, expressions, and anatomical details that would be "
            "appropriate for adult artwork. Use proper anatomical terms and avoid censoring."
        )

        prompt = (
            f"Generate a detailed, descriptive NSFW caption for an image with the following parameters:\n\n"
            f"{param_text}\n\n"
            f"The caption should be detailed and vivid, focusing on visual elements and NSFW content "
            f"appropriate for the specified intensity level. Format your response as a single paragraph "
            f"without explanations or extra text."
        )

        # Generate the caption
        logger.debug(f"Generating NSFW caption for: {subject} with intensity {nsfw_intensity}")
        response = self.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            model=model or self.default_model,
        )

        # Extract and return the caption
        caption = response.get("response", "").strip()
        caption = filter_thinking(caption, show_thinking)
        logger.debug(f"Generated NSFW caption of length: {len(caption)}")

        return caption

    def brainstorm_prompt(
        self,
        concept: str,
        temperature: float = 0.8,
        model: Optional[str] = None,
        show_thinking: bool = False,
    ) -> Dict[str, Any]:
        """Generate prompt variations and suggestions

        Args:
            concept: The concept to brainstorm
            temperature: Temperature for generation (default: 0.8)
            model: The model to use (default: None, uses default_model)
            show_thinking: Whether to include model's thinking process

        Returns:
            Dictionary with prompt variations and suggestions
        """
        system = (
            "You are a creative prompt engineer for text-to-image models. "
            "Your task is to generate variations of a concept for image generation. "
            "For each variation, provide a detailed prompt and relevant tags. "
            "Structure your output as valid JSON."
        )

        prompt = (
            f"Generate 3 creative variations of the following concept for text-to-image generation:\n\n"
            f"Concept: {concept}\n\n"
            f"For each variation, provide:\n"
            f"1. A detailed prompt (1-3 sentences)\n"
            f"2. A list of 5-10 tags that would enhance the image generation\n\n"
            f"Format your response as valid JSON with the following structure:\n"
            f'{{"prompts": [\n'
            f'  {{"prompt": "detailed prompt text", "tags": ["tag1", "tag2", ...]}},\n'
            f'  {{"prompt": "detailed prompt text", "tags": ["tag1", "tag2", ...]}},\n'
            f'  {{"prompt": "detailed prompt text", "tags": ["tag1", "tag2", ...]}}\n'
            f"]}}"
        )

        # Generate the brainstorming response
        logger.debug(f"Brainstorming variations for concept: {concept}")
        response = self.generate(
            prompt=prompt,
            system=system,
            temperature=temperature,
            model=model or self.default_model,
        )

        # Extract and parse the JSON response
        try:
            response_text = response.get("response", "").strip()
            response_text = filter_thinking(response_text, show_thinking)

            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                logger.debug(
                    f"Successfully parsed JSON response with {len(result.get('prompts', []))} variations"
                )
                return result
            else:
                logger.warning("Could not find valid JSON in the response")
                return {"prompts": []}
        except Exception as e:
            logger.error(f"Error parsing JSON response: {str(e)}")
            return {"prompts": []}


# Create a default client for easy import
default_client = OllamaAPIClient()
