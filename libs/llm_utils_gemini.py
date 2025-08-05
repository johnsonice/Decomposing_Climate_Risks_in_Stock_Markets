import os
import json
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from pydantic import BaseModel, Field
import google.generativeai as genai

class GeminiAgent:
    """Base agent class for Google Gemini API interactions."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-pro",
        temperature: float = 0.0,
        max_tokens: int = 4000
    ):
        """
        Initialize the GeminiAgent.
        
        Args:
            api_key: Google API key. If None, uses environment variable
            model: Gemini model identifier to use
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens to generate in response
        """
        # Configure the API key
        genai.configure(api_key=api_key or os.environ.get('GOOGLE_API_KEY'))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.message_history: List[Dict[str, str]] = []
    
    def _get_api_response(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> Any:
        """Make an API call to Google Gemini."""
        try:
            # Initialize the model
            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                **kwargs
            }
            
            gemini_model = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config
            )
            
            # Convert from OpenAI format to Gemini format
            gemini_messages = []
            system_content = None
            
            # Extract system message if present and format messages
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                elif msg["role"] == "user":
                    gemini_messages.append({"role": "user", "parts": [msg["content"]]})
                elif msg["role"] == "assistant":
                    gemini_messages.append({"role": "model", "parts": [msg["content"]]})
            
            # Create a chat session
            chat = gemini_model.start_chat(history=gemini_messages)
            
            # If system message exists, prepend it to the first user message
            if system_content and gemini_messages and gemini_messages[0]["role"] == "user":
                first_msg = f"System instruction: {system_content}\n\nUser message: {gemini_messages[0]['parts'][0]}"
                response = chat.send_message(first_msg)
            else:
                # Get the last user message
                last_user_msg = next((msg["parts"][0] for msg in reversed(gemini_messages) 
                                     if msg["role"] == "user"), "")
                response = chat.send_message(last_user_msg)
            
            return response.text
            
        except Exception as e:
            logging.error(f"Error in Gemini API call: {str(e)}")
            raise
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate a response using the Gemini model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Override default model if provided
            temperature: Override default temperature if provided
            max_tokens: Override default max_tokens if provided
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            Generated text response
        """
        model = model or self.model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        response = self._get_api_response(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        # Update message history
        self.message_history.extend(messages)
        self.message_history.append({"role": "assistant", "content": response})
        
        return response
    
    def extract_json(self, text: str) -> Dict:
        """
        Extract JSON from text response.
        
        Args:
            text: Text containing JSON
            
        Returns:
            Parsed JSON as dictionary
        """
        # Find JSON pattern in the text
        json_pattern = r'```json\s*([\s\S]*?)\s*```|{[\s\S]*}'
        match = re.search(json_pattern, text)
        
        if match:
            json_str = match.group(1) if match.group(1) else match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                logging.error(f"Failed to parse JSON: {json_str}")
                raise
        else:
            logging.error(f"No JSON found in response: {text}")
            raise ValueError("No JSON found in response")
