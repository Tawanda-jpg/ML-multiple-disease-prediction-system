from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # -------------------
    # FILE SYSTEM PATHS
    # -------------------
    log_path: Path
    diabetes_model_path: Path
    heart_disease_model_path: Path
    diabetes_dataset_path: Path
    heart_dataset_path: Path
    hyper_params_yaml_path: Path

    # -------------------
    # ML CONFIG
    # -------------------
    api_url: str
    test_size: float = Field(ge=0.0, le=1.0)
    random_state: int = Field(ge=0)

    # -------------------
    # TARGET COLUMNS
    # -------------------
    diabetes_target_column: str
    heart_disease_target_column: str

    # -------------------
    # Pydantic config
    # -------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    # -------------------
    # CLEANING (strips stray quotes from .env values)
    # -------------------
    @field_validator("*", mode="before")
    def strip_quotes(cls, v):
        if isinstance(v, str):
            v = v.strip()
            while (v.startswith('"') and v.endswith('"')) or \
                  (v.startswith("'") and v.endswith("'")):
                v = v[1:-1].strip()
        return v