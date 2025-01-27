from django.db import models
import os

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
