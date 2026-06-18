from src.training.config.settings import Settings

settings = Settings()

log_path = settings.log_path
diabetes_path = settings.diabetes_dataset_path

print(log_path)
print(diabetes_path)