from django.shortcuts import render, redirect
from django.conf import settings
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from django.core.management import call_command
from pathlib import Path
from audio_features.models import AudioFile
from audio_features.extractors import FeatureExtractorRegistry

@require_http_methods(["GET"])
def audio_list(request):
    """View to list audio files."""
    audio_files = AudioFile.objects.all().order_by('-uploaded_at')
    return render(request, 'audio_features/audio_list.html', {
        'audio_files': audio_files
    })

@require_http_methods(["POST"])
def scan_directory(request):
    """View to handle scanning the audio directory for new files."""
    audio_dir = Path(settings.MEDIA_ROOT)
    new_files = 0
    
    for file_path in audio_dir.glob('*.mp3'):
        # Skip if file already exists in database
        if not AudioFile.objects.filter(title=file_path.stem).exists():
            # Store path relative to MEDIA_ROOT
            relative_path = file_path.name  # Just the filename since it's directly in MEDIA_ROOT
            AudioFile.objects.create(
                title=file_path.stem,
                file=relative_path
            )
            new_files += 1
    
    messages.success(request, f'Successfully found {new_files} new audio files.')
    return redirect('audio_features:list')

@require_http_methods(["POST"])
def extract_features(request):
    """View to handle feature extraction for all audio files."""
    try:
        # Use the management command to extract features
        call_command('extract_features')
        messages.success(request, 'Successfully extracted features for all audio files.')
    except Exception as e:
        messages.error(request, f'Error extracting features: {str(e)}')
    
    return redirect('audio_features:list') 