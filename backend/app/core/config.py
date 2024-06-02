from pydantic import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql://user:password@localhost/air_quality_data"
    model_path: str = "app/model/model.pth"


settings = Settings()
