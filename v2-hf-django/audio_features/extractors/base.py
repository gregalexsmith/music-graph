from abc import ABC, abstractmethod
from typing import Dict, Any
from django.core.exceptions import ValidationError
from ..models import AudioFile

class FeatureExtractor(ABC):
    """Abstract base class for all feature extractors."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this feature extractor."""
        pass
    
    @abstractmethod
    def extract_features(self, audio_file: AudioFile) -> Dict[str, Any]:
        """Extract features from an audio file.
        
        Args:
            audio_file: The AudioFile model instance to extract features from
            
        Returns:
            Dictionary of features where keys are feature names and values are the extracted features
        """
        pass

class FeatureExtractorRegistry:
    """Registry to manage all feature extractors."""
    _extractors: Dict[str, FeatureExtractor] = {}
    
    @classmethod
    def register(cls, extractor: FeatureExtractor) -> None:
        """Register a new feature extractor."""
        if extractor.name in cls._extractors:
            raise ValidationError(f"Feature extractor '{extractor.name}' is already registered")
        cls._extractors[extractor.name] = extractor
    
    @classmethod
    def get_extractor(cls, name: str) -> FeatureExtractor:
        """Get a feature extractor by name."""
        if name not in cls._extractors:
            raise ValidationError(f"Feature extractor '{name}' not found")
        return cls._extractors[name]
    
    @classmethod
    def list_extractors(cls) -> Dict[str, FeatureExtractor]:
        """List all registered feature extractors."""
        return cls._extractors.copy() 