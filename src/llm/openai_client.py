"""
OpenAI LLM client
"""

from typing import List, Dict, Any, Optional
import os
import sys
import json


class OpenAIClient:
    """OpenAI API client for LLM generation"""
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        prompt_for_key: bool = True,
        use_keyring: bool = True,
        save_to_keyring: bool = False,
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
        # Resolve API key via explicit arg -> env -> keyring -> optional interactive prompt
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key and use_keyring:
            try:
                import keyring  # type: ignore
                resolved_key = keyring.get_password("medical_rag_system", "openai_api_key")
            except Exception:
                pass
        self.api_key = resolved_key
        # Resolve optional base URL and organization for gateways (e.g., CMU AI Gateway)
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.organization = organization or os.getenv("OPENAI_ORGANIZATION")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        self._initialize_client(prompt_for_key=prompt_for_key, use_keyring=use_keyring, save_to_keyring=save_to_keyring)
    
    def _initialize_client(self, prompt_for_key: bool = True, use_keyring: bool = True, save_to_keyring: bool = False):
        """Initialize OpenAI client"""
        if not self.api_key:
            # Optionally prompt the user for a key if running interactively
            if prompt_for_key and sys.stdin and sys.stdin.isatty():
                try:
                    import getpass
                    print("OpenAI API key is required. It will not be printed.")
                    entered = getpass.getpass("Paste OpenAI API key: ")
                    if entered:
                        self.api_key = entered.strip()
                        if save_to_keyring:
                            try:
                                import keyring  # type: ignore
                                keyring.set_password("medical_rag_system", "openai_api_key", self.api_key)
                            except Exception:
                                pass
                except Exception:
                    pass
            if not self.api_key:
                print("Warning: No OpenAI API key provided")
                return
        
        try:
            from openai import OpenAI
            # Initialize basic client; avoid unsupported kwargs
            if self.base_url and self.organization:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, organization=self.organization)
            elif self.base_url:
                self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            elif self.organization:
                self.client = OpenAI(api_key=self.api_key, organization=self.organization)
            else:
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
        # Build context in the requested JSON structure
        docs_obj: Dict[str, Any] = {}
        for i, doc in enumerate(context_documents, 1):
            relevance = doc.get("score")
            if relevance is None:
                relevance = doc.get("dense_score")
            try:
                relevance_val = float(relevance) if relevance is not None else 0.0
            except Exception:
                relevance_val = 0.0

            docs_obj[f"doc{i}"] = {
                "PMID": doc.get("doc_id"),
                "title": doc.get("title", ""),
                "content": doc.get("abstract", ""),
                "relevance_score": relevance_val
            }

        user_part = f"User Prompt: Answer the following question: {query}"
        context_part = "Context Prompt: Here are the documents:\n" + json.dumps(docs_obj, ensure_ascii=False, indent=2)
        full_prompt = user_part + "\n\n" + context_part

        return self.generate(full_prompt, system_prompt=system_prompt)
