from pydantic_settings import BaseSettings

class Settings(BaseSettings):

    log_path: str
    diabetes_dataset_path: str
    heart_dataset_path: str
    diabetes_model_path: str
    heart_disease_model_path: str
    diabetes_target_column: str
    heart_disease_target_column: str
    hyper_params_yaml_path: str
    test_size: float
    random_state: int
    api_url: str

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"
