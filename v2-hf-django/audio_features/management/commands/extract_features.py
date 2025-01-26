from django.core.management.base import BaseCommand
from audio_features.models import AudioFile, AudioFeatures, FeatureExtractorRegistry
from django.db import transaction

class Command(BaseCommand):
    help = 'Extract features from audio files using registered feature extractors'

    def add_arguments(self, parser):
        parser.add_argument(
            '--extractor',
            help='Name of specific extractor to use. If not provided, all registered extractors will be used.'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Re-extract features even if they already exist'
        )

    def handle(self, *args, **options):
        extractor_name = options.get('extractor')
        force = options.get('force', False)
        
        # Get all audio files
        audio_files = AudioFile.objects.all()
        
        if extractor_name:
            extractors = {
                extractor_name: FeatureExtractorRegistry.get_extractor(extractor_name)
            }
        else:
            extractors = FeatureExtractorRegistry.list_extractors()

        print(f"Extractors: {extractors}")
        
        for audio_file in audio_files:
            self.stdout.write(f"Processing {audio_file.title}...")
            
            for name, extractor in extractors.items():
                try:
                    with transaction.atomic():
                        # Check if features already exist
                        features, created = AudioFeatures.objects.get_or_create(
                            audio_file=audio_file,
                            model_name=name
                        )
                        
                        if not created and not force:
                            self.stdout.write(f"  Skipping {name} (already exists)")
                            continue
                        
                        # Extract features
                        self.stdout.write(f"  Extracting features using {name}...")
                        feature_data = extractor.extract_features(audio_file)
                        
                        # Update or create features
                        features.features = feature_data
                        features.save()
                        
                        self.stdout.write(self.style.SUCCESS(f"  ✓ Features extracted using {name}"))
                        
                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(f"  ✗ Error extracting features using {name}: {str(e)}")
                    ) 