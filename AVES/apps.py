from django.apps import AppConfig

class AvesConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'AVES'

    def ready(self):
        import AVES.model_loader
