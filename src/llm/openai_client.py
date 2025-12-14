"""
OpenAI LLM client
"""

from typing import List, Dict, Any, Optional
import os


class OpenAIClient:
    """OpenAI API client for LLM generation"""
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        """
        Initialize OpenAI client
        
        Args:
            model: Model name (e.g., gpt-4, gpt-3.5-turbo)
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client"""
        if not self.api_key:
            print("Warning: No OpenAI API key provided")
            return
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except Exception as e:
            print(f"Warning: Could not initialize OpenAI client: {e}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate response from LLM
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Override default temperature
            max_tokens: Override default max tokens
        
        Returns:
            Generated text
        """
        if self.client is None:
            return "Error: OpenAI client not initialized"
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error generating response: {e}"
    
    def generate_with_context(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        system_prompt: str
    ) -> str:
        """
        Generate response with retrieved context
        
        Args:
            query: User query
            context_documents: Retrieved documents with citations
            system_prompt: System prompt
        
        Returns:
            Generated answer
        """
        # Build context from documents
        context_parts = []
        for i, doc in enumerate(context_documents, 1):
            doc_text = f"[{doc['doc_id']}] {doc.get('title', '')}\n{doc.get('abstract', '')}"
            if doc.get('pub_date'):
                doc_text += f"\nPublished: {doc['pub_date']}"
            context_parts.append(doc_text)
        
        context = "\n\n".join(context_parts)
        
        # Build full prompt
        full_prompt = f"{context}\n\n---\n\nQuestion: {query}\n\nAnswer:"
        
        return self.generate(full_prompt, system_prompt=system_prompt)
