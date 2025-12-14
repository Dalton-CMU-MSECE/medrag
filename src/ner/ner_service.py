"""
NER service for biomedical entity extraction
"""

from typing import List, Dict, Any


class NERService:
    """Biomedical Named Entity Recognition service"""
    
    def __init__(self, model_name: str = "en_core_sci_sm", confidence_threshold: float = 0.7):
        """
        Initialize NER service
        
        Args:
            model_name: SciSpacy model name
            confidence_threshold: Minimum confidence for entity extraction
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.nlp = None
        self._load_model()
    
    def _load_model(self):
        """Load the SciSpacy NER model"""
        try:
            import scispacy
            import spacy
            self.nlp = spacy.load(self.model_name)
        except Exception as e:
            print(f"Warning: Could not load SciSpacy model: {e}")
            print("Using placeholder NER service")
            self.nlp = None
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract biomedical entities from text
        
        Args:
            text: Input text
        
        Returns:
            List of entities with text, type, start, end, confidence
        """
        if self.nlp is None:
            # Placeholder: return empty list if model not loaded
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entity = {
                "text": ent.text,
                "type": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "confidence": 1.0  # SciSpacy doesn't provide confidence scores by default
            }
            
            if entity["confidence"] >= self.confidence_threshold:
                entities.append(entity)
        
        return entities
    
    def extract_medical_terms(self, text: str) -> List[str]:
        """
        Extract medical terms from text (simplified version)
        
        Returns:
            List of medical term strings
        """
        entities = self.extract_entities(text)
        return [ent["text"] for ent in entities]
