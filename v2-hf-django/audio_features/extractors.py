from transformers import pipeline
from .models import FeatureExtractor, AudioFile, FeatureExtractorRegistry
from typing import Dict, Any
import torch

class MusicGenreExtractor(FeatureExtractor):
    """Feature extractor for music genre classification using HuggingFace's pipeline."""
    
    def __init__(self):
        # Using MIT's music genre classification model
        self.model = pipeline(
            "audio-classification",
            model="mit/ast-finetuned-audioset-10-10-0.4593",
            device=0 if torch.cuda.is_available() else -1
        )
    
    @property
    def name(self) -> str:
        return "music_genre"
    
    def extract_features(self, audio_file: AudioFile) -> Dict[str, Any]:
        """Extract genre features from an audio file.
        
        Returns:
            Dictionary containing:
            - genres: List of dictionaries with genre labels and scores
            - top_genre: The highest scoring genre
        """
        # The model expects a path to an audio file
        results = self.model(audio_file.file.path)
        
        # Format the results
        features = {
            "genres": results,
            "top_genre": max(results, key=lambda x: x["score"])["label"]
        }
        
        return features


# Register the extractor when this module is imported
FeatureExtractorRegistry.register(MusicGenreExtractor()) 