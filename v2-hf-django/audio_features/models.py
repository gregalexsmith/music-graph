from django.db import models
import os
from abc import ABC, abstractmethod
from typing import Dict, Any
from django.core.exceptions import ValidationError

class AudioFile(models.Model):
    """Model to store audio files and basic metadata."""
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='audio_files/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.title
    
    @property
    def filename(self):
        return os.path.basename(self.file.name)

class AudioFeatures(models.Model):
    """Model to store extracted features from audio files."""
    audio_file = models.OneToOneField(AudioFile, on_delete=models.CASCADE, related_name='features')
    
    # Store all extracted features in a flexible JSON format
    features = models.JSONField(default=dict)
    model_name = models.CharField(max_length=255, help_text="Name of the model used for feature extraction")
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Features for {self.audio_file.title}"

    class Meta:
        verbose_name_plural = "Audio features"

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
