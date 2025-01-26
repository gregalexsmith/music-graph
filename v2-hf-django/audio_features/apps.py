from django.apps import AppConfig


class AudioFeaturesConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'audio_features'

    def ready(self):
        """Import extractors when the app is ready to ensure they are registered."""
        from . import extractors
